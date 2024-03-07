//===- ExtractCircuits.cpp - Extract quantum ops to circuits ----*- C++ -*-===//
//
// (C) Copyright IBM 2024.
//
// This code is part of Qiskit.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file implements the pass for extracting quantum ops into quir.circuits
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/ExtractCircuits.h"

#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/QCS/IR/QCSOps.h"
#include "Dialect/QUIR/IR/QUIRAttributes.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cassert>
#include <optional>
#include <string>
#include <sys/types.h>
#include <vector>

#define DEBUG_TYPE "ExtractCircuits"

using namespace mlir;
using namespace mlir::quir;

llvm::cl::opt<bool>
    enableCircuits("enable-circuits",
                   llvm::cl::desc("enable extract quir circuits"),
                   llvm::cl::init(false));

// NOLINTNEXTLINE(misc-use-anonymous-namespace)
static std::optional<Operation *> localNextQuantumOpOrNull(Operation *op) {
  Operation *nextOp = op;
  while (nextOp) {
    if (isQuantumOp(nextOp) && nextOp != op)
      return nextOp;
    if (nextOp->hasTrait<::mlir::RegionBranchOpInterface::Trait>()) {
      // control flow found, no next quantum op
      return std::nullopt;
    }
    if (isa<qcs::ParallelControlFlowOp>(nextOp))
      return std::nullopt;
    if (isa<oq3::CBitInsertBitOp>(nextOp))
      return std::nullopt;
    if (isa<quir::SwitchOp>(nextOp))
      return std::nullopt;
    nextOp = nextOp->getNextNode();
  }
  return std::nullopt;
} // localNextQuantumOpOrNull

OpBuilder ExtractCircuitsPass::startCircuit(Location location,
                                            OpBuilder topLevelBuilder) {

  inputTypes.clear();
  inputValues.clear();
  outputTypes.clear();
  outputValues.clear();
  originalResults.clear();
  circuitArguments.clear();
  circuitOperands.clear();
  phyiscalIds.clear();

  std::string const circuitName = "circuit_";
  std::string newName = circuitName + std::to_string(circuitCount++);
  while (circuitOpsMap.contains(newName))
    newName = circuitName + std::to_string(circuitCount++);

  currentCircuitOp =
      topLevelBuilder.create<CircuitOp>(location, newName,
                                        topLevelBuilder.getFunctionType(
                                            /*inputs=*/ArrayRef<Type>(),
                                            /*results=*/ArrayRef<Type>()));
  currentCircuitOp.addEntryBlock();
  circuitOpsMap[newName] = currentCircuitOp;

  currentCircuitOp->setAttr(llvm::StringRef("quir.classicalOnly"),
                            topLevelBuilder.getBoolAttr(false));

  LLVM_DEBUG(llvm::dbgs() << "Start Circuit " << currentCircuitOp.getSymName()
                          << "\n");

  OpBuilder circuitBuilder =
      OpBuilder::atBlockBegin(&currentCircuitOp.getBody().front());
  return circuitBuilder;
}

void ExtractCircuitsPass::addToCircuit(
    Operation *currentOp, OpBuilder circuitBuilder,
    llvm::SmallVector<Operation *> &eraseList) {

  IRMapping mapper;
  // add operands to circuit input list
  for (auto operand : currentOp->getOperands()) {
    auto *defOp = operand.getDefiningOp();
    auto search = circuitOperands.find(defOp);
    uint argumentIndex = 0;
    if (search == circuitOperands.end()) {
      argumentIndex = inputValues.size();
      inputValues.push_back(operand);
      circuitOperands[defOp] = argumentIndex;

      currentCircuitOp.insertArgument(argumentIndex, operand.getType(), {},
                                      currentOp->getLoc());
      if (isa<quir::DeclareQubitOp>(defOp)) {
        auto physicalId = defOp->getAttrOfType<IntegerAttr>("id");
        phyiscalIds.push_back(physicalId.getInt());
        currentCircuitOp.setArgAttrs(
            argumentIndex,
            ArrayRef({NamedAttribute(
                StringAttr::get(&getContext(),
                                mlir::quir::getPhysicalIdAttrName()),
                physicalId)}));
      }
    } else {
      argumentIndex = search->second;
    }

    mapper.map(operand, currentCircuitOp.getArgument(argumentIndex));
  }
  auto *newOp = circuitBuilder.clone(*currentOp, mapper);

  outputTypes.append(newOp->getResultTypes().begin(),
                     newOp->getResultTypes().end());
  outputValues.append(newOp->getResults().begin(), newOp->getResults().end());
  originalResults.append(currentOp->getResults().begin(),
                         currentOp->getResults().end());

  eraseList.push_back(currentOp);
}

void ExtractCircuitsPass::endCircuit(
    Operation *firstOp, Operation *lastOp, OpBuilder topLevelBuilder,
    OpBuilder circuitBuilder, llvm::SmallVector<Operation *> &eraseList) {

  LLVM_DEBUG(llvm::dbgs() << "Ending circuit " << currentCircuitOp.getSymName()
                          << "\n");

  circuitBuilder.create<mlir::quir::ReturnOp>(lastOp->getLoc(), outputValues);

  // change the input / output types for the quir.circuit
  auto opType = currentCircuitOp.getFunctionType();
  currentCircuitOp.setType(topLevelBuilder.getFunctionType(
      /*inputs=*/opType.getInputs(),
      /*results=*/ArrayRef<Type>(outputTypes)));

  std::sort(phyiscalIds.begin(), phyiscalIds.end());
  currentCircuitOp->setAttr(
      mlir::quir::getPhysicalIdsAttrName(),
      topLevelBuilder.getI32ArrayAttr(ArrayRef<int>(phyiscalIds)));

  // insert call_circuit
  // NOLINTNEXTLINE(misc-const-correctness)
  OpBuilder builder(firstOp);
  newCallCircuitOp = builder.create<mlir::quir::CallCircuitOp>(
      currentCircuitOp->getLoc(), currentCircuitOp.getSymName(),
      TypeRange(outputTypes), ValueRange(inputValues));

  // remap uses
  assert(originalResults.size() == newCallCircuitOp->getNumResults() &&
         "number of results does not match");
  for (uint cnt = 0; cnt < newCallCircuitOp->getNumResults(); cnt++) {
    originalResults[cnt].replaceAllUsesWith(newCallCircuitOp->getResult(cnt));
    assert(originalResults[cnt].use_empty() && "usage expected to be empty");
  }

  // erase operations
  while (!eraseList.empty()) {
    auto *op = eraseList.back();
    eraseList.pop_back();
    assert(op->use_empty() && "operation usage expected to be empty");
    LLVM_DEBUG(llvm::dbgs() << "Erasing: ");
    LLVM_DEBUG(op->dump());
    op->erase();
  }
}

void ExtractCircuitsPass::processOps(Operation *currentOp,
                                     OpBuilder topLevelBuilder,
                                     OpBuilder circuitBuilder) {

  llvm::SmallVector<Operation *> eraseList;

  Operation *firstQuantumOp = nullptr;

  // Handle Shot Loop delay differently
  if (isa<quir::DelayOp>(currentOp) &&
      isa<qcs::ShotInitOp>(currentOp->getNextNode())) {
    // skip past shot init
    currentOp = currentOp->getNextNode()->getNextNode();
  }

  while (currentOp) {

    // Walk through current block of operations and pull out quantum
    // operations into quir.circuits:
    //
    // 1. Identify first quantum operation
    // 2. Start new circuit and clone quantum operation into circuit
    // 2.a. startCircuit will create a new unique quir.circuit
    // 3. Walk forward node by node
    // 4. If node is a quantum operation clone into circuit
    // 5. If not quantum or if control flow - end circuit
    // 5.a. endCircuit will finish circuit, adjust circuit input / output,
    //      create call_circuit and erase original operations
    // 6. If control flow - recursively call processOps for each region of
    //    control flow

    // do not assume first operation is quantum and find first quantum operation
    if (!firstQuantumOp) {

      if (isQuantumOp(currentOp)) {
        firstQuantumOp = currentOp;
      } else {
        // walk forward for first quantum operation or control flow
        auto firstOrNull = localNextQuantumOpOrNull(currentOp);
        if (firstOrNull) {
          currentOp = firstOrNull.value();
          firstQuantumOp = currentOp;
        }
      }
      if (firstQuantumOp)
        circuitBuilder =
            startCircuit(firstQuantumOp->getLoc(), topLevelBuilder);
    }

    // if operation is a quantum operation clone into circuit
    if (isQuantumOp(currentOp))
      addToCircuit(currentOp, circuitBuilder, eraseList);

    // walk forward for next operation
    auto nextOpOrNull = localNextQuantumOpOrNull(currentOp);
    if (nextOpOrNull) {
      currentOp = nextOpOrNull.value();
      continue;
    }

    // next operation was not quantum so if there is a firstQuantumOp there is
    // an in progress circuit to ben ended.
    if (firstQuantumOp) {
      Operation *lastOp = currentOp;
      // nextOpOrNull was null so advance one node
      currentOp = currentOp->getNextNode();
      endCircuit(firstQuantumOp, lastOp, topLevelBuilder, circuitBuilder,
                 eraseList);
    }
    firstQuantumOp = nullptr;

    if (!currentOp)
      break;

    // handle control flow -- and recursively call processOps for control flow
    // regions

    if (isa<scf::IfOp>(currentOp)) {
      auto ifOp = static_cast<scf::IfOp>(currentOp);
      if (!ifOp.getThenRegion().empty())
        processOps(&ifOp.getThenRegion().front().front(), topLevelBuilder,
                   circuitBuilder);
      if (!ifOp.getElseRegion().empty())
        processOps(&ifOp.getElseRegion().front().front(), topLevelBuilder,
                   circuitBuilder);
    } else if (isa<scf::ForOp>(currentOp)) {
      auto forOp = static_cast<scf::ForOp>(currentOp);
      processOps(&forOp.getBody()->front(), topLevelBuilder, circuitBuilder);
    } else if (isa<scf::WhileOp>(currentOp)) {
      auto whileOp = static_cast<scf::WhileOp>(currentOp);
      if (!whileOp.getBefore().empty())
        processOps(&whileOp.getBefore().front().front(), topLevelBuilder, circuitBuilder);
      if (!whileOp.getAfter().empty())
        processOps(&whileOp.getAfter().front().front(), topLevelBuilder, circuitBuilder);
    } else if (isa<quir::SwitchOp>(currentOp)) {
      // NOLINTNEXTLINE(llvm-qualified-auto)
      auto switchOp = static_cast<quir::SwitchOp>(currentOp);
      for (auto &region : switchOp.getCaseRegions())
        processOps(&region.front().front(), topLevelBuilder, circuitBuilder);
    } else if (isa<qcs::ParallelControlFlowOp>(currentOp)) {
      // NOLINTNEXTLINE(llvm-qualified-auto)
      auto parOp = static_cast<qcs::ParallelControlFlowOp>(currentOp);
      processOps(&parOp.getBody()->front(), topLevelBuilder, circuitBuilder);
    } else if (currentOp->hasTrait<::mlir::RegionBranchOpInterface::Trait>()) {
      currentOp->dump();
      assert(false && "Unhandled control flow");
    }
    currentOp = currentOp->getNextNode();
  }

  // if firstQuantumOp is still set then there is an in progress circuit to be
  // ended
  if (firstQuantumOp) {
    endCircuit(firstQuantumOp, currentOp, topLevelBuilder, circuitBuilder,
               eraseList);
  }
}

void ExtractCircuitsPass::runOnOperation() {
  // do nothing if circuits is not enabled
  if (!enableCircuits)
    return;

  circuitCount = 0;
  currentCircuitOp = nullptr;

  Operation *moduleOp = getOperation();

  llvm::StringMap<Operation *> circuitOpsMap;

  moduleOp->walk([&](CircuitOp circuitOp) {
    circuitOpsMap[circuitOp.getSymName()] = circuitOp.getOperation();
  });

  mlir::func::FuncOp mainFunc =
      dyn_cast<mlir::func::FuncOp>(quir::getMainFunction(moduleOp));
  assert(mainFunc && "could not find the main func");

  auto const builder = OpBuilder(mainFunc);
  auto *firstOp = &mainFunc.getBody().front().front();
  processOps(firstOp, builder, builder);
} // runOnOperation

llvm::StringRef ExtractCircuitsPass::getArgument() const {
  return "extract-circuits";
}
llvm::StringRef ExtractCircuitsPass::getDescription() const {
  return "Extract quantum operations to circuits ";
}

llvm::StringRef ExtractCircuitsPass::getName() const {
  return "Extract Circuits Pass";
}
