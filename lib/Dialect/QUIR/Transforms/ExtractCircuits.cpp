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
#include "Utils/SymbolCacheAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cassert>
#include <string>
#include <sys/types.h>
#include <vector>

#define DEBUG_TYPE "ExtractCircuits"

using namespace mlir;
using namespace mlir::quir;

llvm::cl::opt<bool>
    enableCircuits("enable-circuits",
                   llvm::cl::desc("enable extract quir circuits"),
                   llvm::cl::init(true));

static bool terminatesCircuit(Operation &op) {
  return (op.hasTrait<::mlir::RegionBranchOpInterface::Trait>() ||
          isa<qcs::ParallelControlFlowOp>(op) ||
          isa<oq3::CBitInsertBitOp>(op) || isa<quir::SwitchOp>(op) ||
          isa<qcs::SynchronizeOp>(op));
} // terminatesCircuit

OpBuilder ExtractCircuitsPass::startCircuit(Location location,
                                            OpBuilder topLevelBuilder) {

  inputTypes.clear();
  inputValues.clear();
  outputTypes.clear();
  outputValues.clear();
  originalResults.clear();
  circuitOperands.clear();
  phyiscalIds.clear();
  argToId.clear();

  std::string const circuitName = "circuit_";
  std::string newName = circuitName + std::to_string(circuitCount++);
  assert(symbolCache && "symbolCache not set");
  while (symbolCache->contains(newName))
    newName = circuitName + std::to_string(circuitCount++);

  currentCircuitOp =
      topLevelBuilder.create<CircuitOp>(location, newName,
                                        topLevelBuilder.getFunctionType(
                                            /*inputs=*/ArrayRef<Type>(),
                                            /*results=*/ArrayRef<Type>()));
  currentCircuitOp.addEntryBlock();
  symbolCache->addCallee(currentCircuitOp);

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
    mlir::Value mappedValue;
    if (search == circuitOperands.end()) {
      // Check if we should embed in the circuit
      auto constantLike = (isa<mlir::quir::ConstantOp>(defOp) ||
                           isa<qcs::ParameterLoadOp>(defOp));
      if (constantLike && !mapper.contains(operand)) {
        auto *newDefOp = circuitBuilder.clone(*defOp, mapper);
        mappedValue = newDefOp->getResult(0);
        eraseList.push_back(defOp);
      } else {
        // Otherwise we add to the circuit signature
        argumentIndex = inputValues.size();
        inputValues.push_back(operand);
        inputTypes.push_back(operand.getType());
        circuitOperands[defOp] = argumentIndex;
        currentCircuitOp.getBody().addArgument(operand.getType(),
                                               currentOp->getLoc());

        if (isa<quir::DeclareQubitOp>(defOp)) {
          auto id = defOp->getAttrOfType<IntegerAttr>("id").getInt();
          phyiscalIds.push_back(id);
          argToId[argumentIndex] = id;
        }
        mappedValue = currentCircuitOp.getArgument(argumentIndex);
      }
    } else {
      argumentIndex = search->second;
      mappedValue = currentCircuitOp.getArgument(argumentIndex);
    }
    mapper.map(operand, mappedValue);
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
      /*inputs=*/ArrayRef<Type>(inputTypes),
      /*results=*/ArrayRef<Type>(outputTypes)));

  for (const auto &[key, value] : argToId)
    currentCircuitOp.setArgAttrs(
        key,
        ArrayRef({NamedAttribute(
            StringAttr::get(&getContext(), mlir::quir::getPhysicalIdAttrName()),
            topLevelBuilder.getI32IntegerAttr(value))}));

  std::sort(phyiscalIds.begin(), phyiscalIds.end());
  currentCircuitOp->setAttr(
      mlir::quir::getPhysicalIdsAttrName(),
      topLevelBuilder.getI32ArrayAttr(ArrayRef<int>(phyiscalIds)));

  // insert call_circuit
  // NOLINTNEXTLINE(misc-const-correctness)
  OpBuilder builder(lastOp);
  builder.setInsertionPointAfter(lastOp);
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

  currentCircuitOp = nullptr;
}

void ExtractCircuitsPass::processRegion(mlir::Region &region,
                                        OpBuilder topLevelBuilder,
                                        OpBuilder circuitBuilder) {
  for (mlir::Block &block : region.getBlocks())
    processBlock(block, topLevelBuilder, circuitBuilder);
}

void ExtractCircuitsPass::processBlock(mlir::Block &block,
                                       OpBuilder topLevelBuilder,
                                       OpBuilder circuitBuilder) {
  llvm::SmallVector<Operation *> eraseList;
  Operation *firstQuantumOp = nullptr;
  Operation *lastQuantumOp = nullptr;

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
  // 6. If control flow - recursively call processRegion for each region of
  //    control flow
  for (Operation &currentOp : llvm::make_early_inc_range(block)) {
    // Handle Shot Loop delay differently
    if (isa<quir::DelayOp>(currentOp) &&
        isa<qcs::ShotInitOp>(currentOp.getNextNode())) {
      // skip past shot init
      continue;
    }
    if (isQuantumOp(&currentOp)) {
      // Start building circuit if not already
      lastQuantumOp = &currentOp;
      if (!currentCircuitOp) {
        firstQuantumOp = lastQuantumOp;
        circuitBuilder =
            startCircuit(firstQuantumOp->getLoc(), topLevelBuilder);
      }
      addToCircuit(&currentOp, circuitBuilder, eraseList);
      continue;
    }
    if (terminatesCircuit(currentOp)) {
      // next operation was not quantum so if there is a circuit builder in
      // progress there is an in progress circuit to be ended.
      if (currentCircuitOp) {
        endCircuit(firstQuantumOp, lastQuantumOp, topLevelBuilder,
                   circuitBuilder, eraseList);
      }

      // handle control flow by recursively calling processBlock for control
      // flow regions
      for (mlir::Region &region : currentOp.getRegions())
        processRegion(region, topLevelBuilder, circuitBuilder);
    }
  }
  // End of block complete the circuit
  if (currentCircuitOp) {
    endCircuit(firstQuantumOp, lastQuantumOp, topLevelBuilder, circuitBuilder,
               eraseList);
  }
}

void ExtractCircuitsPass::runOnOperation() {
  // do nothing if circuits is not enabled
  if (!enableCircuits)
    return;

  Operation *moduleOp = getOperation();

  symbolCache =
      &getAnalysis<qssc::utils::SymbolCacheAnalysis>().addToCache<CircuitOp>();

  mlir::func::FuncOp mainFunc =
      dyn_cast<mlir::func::FuncOp>(quir::getMainFunction(moduleOp));
  assert(mainFunc && "could not find the main func");

  auto const builder = OpBuilder(mainFunc);
  processRegion(mainFunc.getRegion(), builder, builder);
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
