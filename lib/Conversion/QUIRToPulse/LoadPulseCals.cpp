//===- LoadPulseCals.cpp ----------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
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
/// This file implements the pass to load the pulse calibrations.
///
//===----------------------------------------------------------------------===//

#include "Conversion/QUIRToPulse/LoadPulseCals.h"

#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/Pulse/IR/PulseOps.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTraits.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <sys/types.h>
#include <utility>
#include <vector>

#define DEBUG_TYPE "LoadPulseCalsDebug"

using namespace mlir;
using namespace mlir::quir;
using namespace mlir::pulse;

void LoadPulseCalsPass::runOnOperation() {

  mlir::ModuleOp const moduleOp = getOperation();
  mlir::func::FuncOp mainFunc =
      dyn_cast<mlir::func::FuncOp>(quir::getMainFunction(moduleOp));
  assert(mainFunc && "could not find the main func");

  // check for command line override of the path to default pulse cals
  if (defaultPulseCals.hasValue())
    DEFAULT_PULSE_CALS = defaultPulseCals.getValue();

  // parse the default pulse calibrations
  if (!DEFAULT_PULSE_CALS.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "parsing default pulse calibrations.\n");
    if (auto err = parsePulseCalsModuleOp(DEFAULT_PULSE_CALS,
                                          defaultPulseCalsModule)) {
      llvm::dbgs() << err;
      return signalPassFailure();
    }
    // add sequence Ops to pulseCalsNameToSequenceMap
    defaultPulseCalsModule->walk([&](mlir::pulse::SequenceOp sequenceOp) {
      auto sequenceName = sequenceOp.getSymName().str();
      pulseCalsNameToSequenceMap[sequenceName] = sequenceOp;
    });
  } else
    LLVM_DEBUG(llvm::dbgs()
               << "default pulse calibrations path is not specified.\n");

  // parse the additional pulse calibrations
  if (!ADDITIONAL_PULSE_CALS.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "parsing additional pulse calibrations.\n");
    if (auto err = parsePulseCalsModuleOp(ADDITIONAL_PULSE_CALS,
                                          additionalPulseCalsModule)) {
      llvm::dbgs() << err;
      return signalPassFailure();
    }
    // add sequence Ops to pulseCalsNameToSequenceMap
    additionalPulseCalsModule->walk([&](mlir::pulse::SequenceOp sequenceOp) {
      auto sequenceName = sequenceOp.getSymName().str();
      pulseCalsNameToSequenceMap[sequenceName] = sequenceOp;
    });
  } else
    LLVM_DEBUG(llvm::dbgs()
               << "additional pulse calibrations path is not specified.\n");

  // parse the user specified pulse calibrations
  LLVM_DEBUG(llvm::dbgs() << "parsing user specified pulse calibrations.\n");
  moduleOp->walk([&](mlir::pulse::SequenceOp sequenceOp) {
    auto sequenceName = sequenceOp.getSymName().str();
    pulseCalsNameToSequenceMap[sequenceName] = sequenceOp;
    pulseCalsAddedToIR.insert(sequenceName);
  });

  moduleOp->walk(
      [&](CallCircuitOp callCircOp) { loadPulseCals(callCircOp, mainFunc); });
}

void LoadPulseCalsPass::loadPulseCals(CallCircuitOp callCircuitOp,
                                      mlir::func::FuncOp funcOp) {

  auto circuitOp = getCircuitOp(callCircuitOp);
  circuitOp->walk([&](Operation *op) {
    if (auto castOp = dyn_cast<CallCircuitOp>(op))
      llvm_unreachable(
          "CallCircuitOp inside another CircuitOp is not allowed in this pass");
    else if (auto castOp = dyn_cast<CallGateOp>(op))
      loadPulseCals(castOp, callCircuitOp, funcOp);
    else if (auto castOp = dyn_cast<BuiltinCXOp>(op))
      loadPulseCals(castOp, callCircuitOp, funcOp);
    else if (auto castOp = dyn_cast<Builtin_UOp>(op))
      loadPulseCals(castOp, callCircuitOp, funcOp);
    else if (auto castOp = dyn_cast<MeasureOp>(op))
      loadPulseCals(castOp, callCircuitOp, funcOp);
    else if (auto castOp = dyn_cast<mlir::quir::BarrierOp>(op))
      loadPulseCals(castOp, callCircuitOp, funcOp);
    else if (auto castOp = dyn_cast<mlir::quir::DelayOp>(op))
      loadPulseCals(castOp, callCircuitOp, funcOp);
    else if (auto castOp = dyn_cast<mlir::quir::ResetQubitOp>(op))
      loadPulseCals(castOp, callCircuitOp, funcOp);
    else {
      LLVM_DEBUG(llvm::dbgs() << "no pulse cal loading needed for " << op);
      assert((!op->hasTrait<mlir::quir::UnitaryOp>() and
              !op->hasTrait<mlir::quir::CPTPOp>()) &&
             "unkown operation");
    }
  });
}

void LoadPulseCalsPass::loadPulseCals(CallGateOp callGateOp,
                                      CallCircuitOp callCircuitOp,
                                      mlir::func::FuncOp funcOp) {
  std::vector<Value> qubitOperands;
  qubitCallOperands(callGateOp, qubitOperands);
  std::vector<uint32_t> qubits = getQubitOperands(qubitOperands, callCircuitOp);
  std::string gateName = callGateOp.getCalleeAttr().getValue().str();
  std::string const gateMangledName = getMangledName(gateName, qubits);
  assert(pulseCalsNameToSequenceMap.find(gateMangledName) !=
             pulseCalsNameToSequenceMap.end() &&
         "could not find any pulse calibration for call gate");

  OpBuilder builder = OpBuilder::atBlockBegin(&funcOp.getBody().front());
  callGateOp->setAttr("pulse.calName", builder.getStringAttr(gateMangledName));
  addPulseCalToModule(funcOp, pulseCalsNameToSequenceMap[gateMangledName]);
}

void LoadPulseCalsPass::loadPulseCals(BuiltinCXOp CXOp,
                                      CallCircuitOp callCircuitOp,
                                      mlir::func::FuncOp funcOp) {

  std::vector<Value> qubitOperands;
  qubitOperands.push_back(CXOp.getControl());
  qubitOperands.push_back(CXOp.getTarget());
  std::vector<uint32_t> qubits = getQubitOperands(qubitOperands, callCircuitOp);
  std::string gateName = "cx";
  std::string const gateMangledName = getMangledName(gateName, qubits);
  assert(pulseCalsNameToSequenceMap.find(gateMangledName) !=
             pulseCalsNameToSequenceMap.end() &&
         "could not find any pulse calibration for the CX gate");

  OpBuilder builder = OpBuilder::atBlockBegin(&funcOp.getBody().front());
  CXOp->setAttr("pulse.calName", builder.getStringAttr(gateMangledName));
  addPulseCalToModule(funcOp, pulseCalsNameToSequenceMap[gateMangledName]);
}

void LoadPulseCalsPass::loadPulseCals(Builtin_UOp UOp,
                                      CallCircuitOp callCircuitOp,
                                      mlir::func::FuncOp funcOp) {

  std::vector<Value> qubitOperands;
  qubitOperands.push_back(UOp.getTarget());
  std::vector<uint32_t> qubits = getQubitOperands(qubitOperands, callCircuitOp);
  std::string gateName = "u3";
  std::string const gateMangledName = getMangledName(gateName, qubits);
  assert(pulseCalsNameToSequenceMap.find(gateMangledName) !=
             pulseCalsNameToSequenceMap.end() &&
         "could not find any pulse calibration for the U gate");

  OpBuilder builder = OpBuilder::atBlockBegin(&funcOp.getBody().front());
  UOp->setAttr("pulse.calName", builder.getStringAttr(gateMangledName));
  addPulseCalToModule(funcOp, pulseCalsNameToSequenceMap[gateMangledName]);
}

void LoadPulseCalsPass::loadPulseCals(MeasureOp measureOp,
                                      CallCircuitOp callCircuitOp,
                                      mlir::func::FuncOp funcOp) {

  OpBuilder builder = OpBuilder::atBlockBegin(&funcOp.getBody().front());

  std::vector<Value> qubitOperands;
  qubitCallOperands<MeasureOp>(measureOp, qubitOperands);
  std::vector<uint32_t> qubits = getQubitOperands(qubitOperands, callCircuitOp);
  std::string gateName = "measure";
  // check if the measurement is marked with quir.midCircuitMeasure
  if (measureOp->hasAttr("quir.midCircuitMeasure"))
    gateName = "mid_circuit_measure";
  std::string const gateMangledName = getMangledName(gateName, qubits);
  measureOp->setAttr("pulse.calName", builder.getStringAttr(gateMangledName));
  if (pulseCalsNameToSequenceMap.find(gateMangledName) !=
      pulseCalsNameToSequenceMap.end()) {
    // found a pulse calibration for the measurement gate
    addPulseCalToModule(funcOp, pulseCalsNameToSequenceMap[gateMangledName]);
    return;
  }
  // did not find a pulse calibration for the gate
  // check if there exists pulse calibrations for individual qubits, and if
  // yes, merge them and add the merged pulse sequence to the module
  std::vector<SequenceOp> sequenceOps;
  for (const auto &qubit : qubits) {
    std::string const individualGateMangledName =
        getMangledName(gateName, qubit);
    assert(pulseCalsNameToSequenceMap.find(individualGateMangledName) !=
               pulseCalsNameToSequenceMap.end() &&
           "could not find pulse calibrations for the measurement gate");
    sequenceOps.push_back(
        pulseCalsNameToSequenceMap[individualGateMangledName]);
  }
  SequenceOp const mergedPulseSequenceOp =
      mergePulseSequenceOps(sequenceOps, gateMangledName);
  pulseCalsNameToSequenceMap[gateMangledName] = mergedPulseSequenceOp;
  addPulseCalToModule(funcOp, mergedPulseSequenceOp);
}

void LoadPulseCalsPass::loadPulseCals(mlir::quir::BarrierOp barrierOp,
                                      CallCircuitOp callCircuitOp,
                                      mlir::func::FuncOp funcOp) {

  OpBuilder builder = OpBuilder::atBlockBegin(&funcOp.getBody().front());

  std::vector<Value> qubitOperands;
  qubitCallOperands<mlir::quir::BarrierOp>(barrierOp, qubitOperands);
  std::vector<uint32_t> qubits = getQubitOperands(qubitOperands, callCircuitOp);
  std::string gateName = "barrier";
  std::string const gateMangledName = getMangledName(gateName, qubits);
  barrierOp->setAttr("pulse.calName", builder.getStringAttr(gateMangledName));
  if (pulseCalsNameToSequenceMap.find(gateMangledName) !=
      pulseCalsNameToSequenceMap.end()) {
    // found a pulse calibration for the barrier gate
    addPulseCalToModule(funcOp, pulseCalsNameToSequenceMap[gateMangledName]);
    return;
  }
  // did not find a pulse calibration for the gate
  // check if there exists pulse calibrations for individual qubits, and if
  // yes, merge them and add the merged pulse sequence to the module
  std::vector<SequenceOp> sequenceOps;
  for (const auto &qubit : qubits) {
    std::string const individualGateMangledName =
        getMangledName(gateName, qubit);
    assert(pulseCalsNameToSequenceMap.find(individualGateMangledName) !=
               pulseCalsNameToSequenceMap.end() &&
           "could not find pulse calibrations for the barrier gate");
    sequenceOps.push_back(
        pulseCalsNameToSequenceMap[individualGateMangledName]);
  }
  SequenceOp const mergedPulseSequenceOp =
      mergePulseSequenceOps(sequenceOps, gateMangledName);
  pulseCalsNameToSequenceMap[gateMangledName] = mergedPulseSequenceOp;
  mergedPulseSequenceOp->setAttr("pulse.duration",
                                 builder.getI64IntegerAttr(0));
  addPulseCalToModule(funcOp, mergedPulseSequenceOp);
}

void LoadPulseCalsPass::loadPulseCals(mlir::quir::DelayOp delayOp,
                                      CallCircuitOp callCircuitOp,
                                      mlir::func::FuncOp funcOp) {

  OpBuilder builder = OpBuilder::atBlockBegin(&funcOp.getBody().front());

  std::vector<Value> qubitOperands;
  qubitCallOperands<mlir::quir::DelayOp>(delayOp, qubitOperands);
  std::vector<uint32_t> qubits = getQubitOperands(qubitOperands, callCircuitOp);
  std::string gateName = "delay";
  std::string const gateMangledName = getMangledName(gateName, qubits);
  delayOp->setAttr("pulse.calName", builder.getStringAttr(gateMangledName));
  if (pulseCalsNameToSequenceMap.find(gateMangledName) !=
      pulseCalsNameToSequenceMap.end()) {
    // found a pulse calibration for the delay gate
    addPulseCalToModule(funcOp, pulseCalsNameToSequenceMap[gateMangledName]);
    return;
  }
  // did not find a pulse calibration for the gate
  // check if there exists pulse calibrations for individual qubits, and if
  // yes, merge them and add the merged pulse sequence to the module
  std::vector<SequenceOp> sequenceOps;
  for (const auto &qubit : qubits) {
    std::string const individualGateMangledName =
        getMangledName(gateName, qubit);
    assert(pulseCalsNameToSequenceMap.find(individualGateMangledName) !=
               pulseCalsNameToSequenceMap.end() &&
           "could not find pulse calibrations for the delay gate");
    sequenceOps.push_back(
        pulseCalsNameToSequenceMap[individualGateMangledName]);
  }
  SequenceOp const mergedPulseSequenceOp =
      mergePulseSequenceOps(sequenceOps, gateMangledName);
  removeRedundantDelayArgs(mergedPulseSequenceOp, builder);
  pulseCalsNameToSequenceMap[gateMangledName] = mergedPulseSequenceOp;
  addPulseCalToModule(funcOp, mergedPulseSequenceOp);
}

void LoadPulseCalsPass::loadPulseCals(mlir::quir::ResetQubitOp resetOp,
                                      CallCircuitOp callCircuitOp,
                                      mlir::func::FuncOp funcOp) {

  OpBuilder builder = OpBuilder::atBlockBegin(&funcOp.getBody().front());

  std::vector<Value> qubitOperands;
  qubitCallOperands<mlir::quir::ResetQubitOp>(resetOp, qubitOperands);
  std::vector<uint32_t> qubits = getQubitOperands(qubitOperands, callCircuitOp);
  std::string gateName = "reset";
  std::string const gateMangledName = getMangledName(gateName, qubits);
  resetOp->setAttr("pulse.calName", builder.getStringAttr(gateMangledName));
  if (pulseCalsNameToSequenceMap.find(gateMangledName) !=
      pulseCalsNameToSequenceMap.end()) {
    // found a pulse calibration for the gate
    addPulseCalToModule(funcOp, pulseCalsNameToSequenceMap[gateMangledName]);
    return;
  }
  // did not find a pulse calibration for the gate
  // check if there exists pulse calibrations for individual qubits, and if
  // yes, merge them and add the merged pulse sequence to the module
  std::vector<SequenceOp> sequenceOps;
  for (const auto &qubit : qubits) {
    std::string const individualGateMangledName =
        getMangledName(gateName, qubit);
    assert(pulseCalsNameToSequenceMap.find(individualGateMangledName) !=
               pulseCalsNameToSequenceMap.end() &&
           "could not find pulse calibrations for the reset gate");
    sequenceOps.push_back(
        pulseCalsNameToSequenceMap[individualGateMangledName]);
  }
  SequenceOp const mergedPulseSequenceOp =
      mergePulseSequenceOps(sequenceOps, gateMangledName);
  pulseCalsNameToSequenceMap[gateMangledName] = mergedPulseSequenceOp;
  addPulseCalToModule(funcOp, mergedPulseSequenceOp);
}

void LoadPulseCalsPass::addPulseCalToModule(
    mlir::func::FuncOp funcOp, mlir::pulse::SequenceOp sequenceOp) {
  if (pulseCalsAddedToIR.find(sequenceOp.getSymName().str()) ==
      pulseCalsAddedToIR.end()) {
    OpBuilder builder = OpBuilder::atBlockBegin(&funcOp.getBody().front());
    auto *clonedPulseCalOp = builder.clone(*sequenceOp);
    auto clonedPulseCalSequenceOp = static_cast<SequenceOp>(clonedPulseCalOp);
    clonedPulseCalSequenceOp->moveBefore(funcOp);
    pulseCalsAddedToIR.insert(sequenceOp.getSymName().str());
  } else
    LLVM_DEBUG(llvm::dbgs() << "pulse cal " << sequenceOp.getSymName().str()
                            << " is already added to IR.\n");
}

llvm::Error LoadPulseCalsPass::parsePulseCalsModuleOp(
    std::string &pulseCalsPath,
    mlir::OwningOpRef<mlir::ModuleOp> &owningOpRef) {
  std::string errorMessage;
  llvm::SourceMgr sourceMgr;
  std::unique_ptr<llvm::MemoryBuffer> pulseCalsFile =
      mlir::openInputFile(pulseCalsPath, &errorMessage);
  if (!pulseCalsFile)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to open pulse calibrations file: " +
                                       errorMessage);
  sourceMgr.AddNewSourceBuffer(std::move(pulseCalsFile), llvm::SMLoc());
  owningOpRef = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &getContext());
  if (!owningOpRef)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to parse pulse calibrations file: " +
                                       pulseCalsPath);
  return llvm::Error::success();
}

mlir::pulse::SequenceOp LoadPulseCalsPass::mergePulseSequenceOps(
    std::vector<mlir::pulse::SequenceOp> &sequenceOps,
    const std::string &mergedSequenceOpName) {

  assert(sequenceOps.size() && "sequence op vector is empty; nothing to merge");

  SequenceOp const firstSequenceOp = sequenceOps[0];

  OpBuilder builder(firstSequenceOp);

  llvm::SmallVector<Type> inputTypes;
  llvm::SmallVector<Value> const inputValues;
  llvm::SmallVector<Type> outputTypes;
  llvm::SmallVector<Value> outputValues;

  // merge input type into single SmallVector
  for (auto sequenceOp : sequenceOps)
    inputTypes.append(sequenceOp->getOperandTypes().begin(),
                      sequenceOp->getOperandTypes().end());

  // create new merged sequence op by cloning the first sequence op
  SequenceOp mergedSequenceOp =
      cast<SequenceOp>(builder.clone(*firstSequenceOp));
  mergedSequenceOp->setAttr(
      SymbolTable::getSymbolAttrName(),
      StringAttr::get(firstSequenceOp->getContext(), mergedSequenceOpName));

  // map original arguments for new sequence based on original sequences'
  // argument numbers
  IRMapping mapper;
  auto baseArgNum = mergedSequenceOp.getNumArguments();
  for (std::size_t seqNum = 1; seqNum < sequenceOps.size(); seqNum++) {
    for (uint cnt = 0; cnt < sequenceOps[seqNum].getNumArguments(); cnt++) {
      auto arg = sequenceOps[seqNum].getArgument(cnt);
      auto dictArg = sequenceOps[seqNum].getArgAttrDict(cnt);
      mergedSequenceOp.insertArgument(baseArgNum + cnt, arg.getType(), dictArg,
                                      arg.getLoc());
      mapper.map(arg, mergedSequenceOp.getArgument(baseArgNum + cnt));
    }
    baseArgNum += sequenceOps[seqNum].getNumArguments();
  }

  // clone the body of the original sequence ops
  builder.setInsertionPointAfter(&mergedSequenceOp.back().back());
  for (std::size_t seqNum = 1; seqNum < sequenceOps.size(); seqNum++) {
    for (auto &block : sequenceOps[seqNum].getBody().getBlocks())
      for (auto &op : block.getOperations())
        builder.clone(op, mapper);
  }

  // remove any existing return operations from new merged sequence op
  // collect their output types and values into vectors
  mergedSequenceOp->walk([&](pulse::ReturnOp returnOp) {
    outputValues.append(returnOp.getOperands().begin(),
                        returnOp->getOperands().end());
    outputTypes.append(returnOp->getOperandTypes().begin(),
                       returnOp->getOperandTypes().end());
    returnOp->erase();
  });

  // create a return op in the merged sequence op with the merged output values
  builder.create<pulse::ReturnOp>(mergedSequenceOp.back().back().getLoc(),
                                  outputValues);

  // change the input / output types for the merged sequence op
  auto opType = mergedSequenceOp.getFunctionType();
  mergedSequenceOp.setType(builder.getFunctionType(
      /*inputs=*/opType.getInputs(),
      /*results=*/ArrayRef<Type>(outputTypes)));

  // check if ALL the sequence ops has the same pulse.duration, and if no,
  // remove the pulse.duration from the merged sequence op if there exists any;
  // the pulse.duration of the merged sequence op needs to be re-calculated in
  // this case.
  // If yes, no further action is required because the duration is already
  // cloned when we clone the first sequence
  bool const allSequenceOpsHasSameDuration =
      doAllSequenceOpsHaveSameDuration(sequenceOps);
  if (!allSequenceOpsHasSameDuration and
      mergedSequenceOp->hasAttr("pulse.duration"))
    mergedSequenceOp->removeAttr("pulse.duration");

  // check if ALL the sequence ops has args/argPorts attr, and if yes,
  // merge the attributes and add them to the merged sequence op
  std::vector<mlir::Attribute> pulseSequenceOpArgs;
  std::vector<mlir::Attribute> pulseSequenceOpArgPorts;
  bool const allSequenceOpsHasArgsAttr =
      mergeAttributes(sequenceOps, "pulse.args", pulseSequenceOpArgs);
  bool const allSequenceOpsHasArgPortsAttr =
      mergeAttributes(sequenceOps, "pulse.argPorts", pulseSequenceOpArgPorts);

  if (allSequenceOpsHasArgsAttr) {
    mlir::ArrayAttr const arrayAttr = builder.getArrayAttr(pulseSequenceOpArgs);
    mergedSequenceOp->setAttr("pulse.args", arrayAttr);
  } else if (mergedSequenceOp->hasAttr("pulse.args"))
    mergedSequenceOp->removeAttr("pulse.args");

  if (allSequenceOpsHasArgPortsAttr) {
    mlir::ArrayAttr const arrayAttr =
        builder.getArrayAttr(pulseSequenceOpArgPorts);
    mergedSequenceOp->setAttr("pulse.argPorts", arrayAttr);
  } else if (mergedSequenceOp->hasAttr("pulse.argPorts"))
    mergedSequenceOp->removeAttr("pulse.argPorts");

  return mergedSequenceOp;
}

void LoadPulseCalsPass::removeRedundantDelayArgs(
    mlir::pulse::SequenceOp sequenceOp, mlir::OpBuilder &builder) {

  // find the first delay arg with integer type, and following redundant delay
  // args
  bool delayArgEncountered = false;
  BlockArgument delayArg;
  std::vector<BlockArgument> redundantArgsToRemove;
  std::vector<uint> redundantArgIndicesToRemove;
  for (uint argIndex = 0; argIndex < sequenceOp.getNumArguments(); argIndex++) {
    BlockArgument const arg = sequenceOp.getArgument(argIndex);
    if (arg.getType().isa<IntegerType>()) {
      if (delayArgEncountered) {
        redundantArgsToRemove.push_back(arg);
        redundantArgIndicesToRemove.push_back(argIndex);
      } else {
        delayArgEncountered = true;
        delayArg = arg;
      }
    }
  }
  assert(delayArgEncountered && "no delay arg with integer type exists");

  // need to update pulse.args and pulse.argPorts if it exists
  std::vector<mlir::Attribute> argAttrVec;
  std::vector<mlir::Attribute> argPortsAttrVec;
  bool const sequenceOpHasPulseArgs =
      sequenceOp->hasAttrOfType<ArrayAttr>("pulse.args");
  bool const sequenceOpHasPulseArgPorts =
      sequenceOp->hasAttrOfType<ArrayAttr>("pulse.argPorts");
  if (sequenceOpHasPulseArgs)
    for (auto attr : sequenceOp->getAttrOfType<ArrayAttr>("pulse.args"))
      argAttrVec.push_back(attr);
  if (sequenceOpHasPulseArgPorts)
    for (auto attr : sequenceOp->getAttrOfType<ArrayAttr>("pulse.argPorts"))
      argPortsAttrVec.push_back(attr);

  // replace all uses of the redundant args
  for (auto arg : redundantArgsToRemove)
    arg.replaceAllUsesWith(delayArg);

  // erase the redundant args
  std::sort(redundantArgIndicesToRemove.begin(),
            redundantArgIndicesToRemove.end());
  std::reverse(redundantArgIndicesToRemove.begin(),
               redundantArgIndicesToRemove.end());
  for (auto argIndex : redundantArgIndicesToRemove) {
    sequenceOp.eraseArgument(argIndex);
    // update the pulse.args and pulse.argPorts vectors
    if (sequenceOpHasPulseArgs)
      argAttrVec.erase(argAttrVec.begin() + argIndex);
    if (sequenceOpHasPulseArgPorts)
      argPortsAttrVec.erase(argPortsAttrVec.begin() + argIndex);
  }

  if (!argPortsAttrVec.empty())
    sequenceOp->setAttr("pulse.argPorts",
                        builder.getArrayAttr(argPortsAttrVec));

  if (!argAttrVec.empty())
    sequenceOp->setAttr("pulse.args", builder.getArrayAttr(argAttrVec));
}

bool LoadPulseCalsPass::doAllSequenceOpsHaveSameDuration(
    std::vector<mlir::pulse::SequenceOp> &sequenceOps) {
  bool prevSequenceEncountered = false;
  uint prevSequencePulseDuration = 0;
  for (const auto &sequenceOp : sequenceOps) {
    if (!sequenceOp->hasAttrOfType<IntegerAttr>("pulse.duration"))
      return false;

    uint const sequenceDuration = static_cast<uint64_t>(
        sequenceOp->getAttrOfType<IntegerAttr>("pulse.duration").getInt());
    if (!prevSequenceEncountered) {
      prevSequenceEncountered = true;
      prevSequencePulseDuration = sequenceDuration;
    } else if (sequenceDuration != prevSequencePulseDuration)
      return false;
  }

  return true;
}

bool LoadPulseCalsPass::mergeAttributes(
    std::vector<mlir::pulse::SequenceOp> &sequenceOps,
    const std::string &attrName, std::vector<mlir::Attribute> &attrVector) {

  bool allSequenceOpsHasAttr = true;
  for (const auto &sequenceOp : sequenceOps) {
    if (sequenceOp->hasAttr(attrName)) {
      auto pulseArgs = sequenceOp->getAttrOfType<ArrayAttr>(attrName);
      for (auto arg : pulseArgs)
        attrVector.push_back(arg);
    } else {
      allSequenceOpsHasAttr = false;
      break;
    }
  }
  return allSequenceOpsHasAttr;
}

std::string LoadPulseCalsPass::getMangledName(std::string &gateName,
                                              std::vector<uint32_t> &qubits) {
  std::string gateMangledName = gateName;
  for (int const qubit : qubits) {
    gateMangledName += "_";
    gateMangledName += std::to_string(qubit);
  }
  return gateMangledName;
}

std::string LoadPulseCalsPass::getMangledName(std::string &gateName,
                                              uint32_t qubit) {
  std::string gateMangledName = gateName + "_" + std::to_string(qubit);
  return gateMangledName;
}

std::vector<uint32_t>
LoadPulseCalsPass::getQubitOperands(std::vector<Value> &qubitOperands,
                                    CallCircuitOp callCircuitOp) {

  std::vector<uint32_t> qubits;
  for (auto &qubit : qubitOperands) {
    if (auto declOp = qubit.getDefiningOp<quir::DeclareQubitOp>()) {
      auto qubitId = quir::lookupQubitId(declOp);
      if (qubitId.has_value())
        qubits.push_back(qubitId.value());
      else
        declOp->emitError() << "Could not find qubit id.";
    } else {
      // qubit is a block argument
      auto blockArg = qubit.dyn_cast<BlockArgument>();
      unsigned const argIdx = blockArg.getArgNumber();
      auto qubitOperand = callCircuitOp->getOperand(argIdx);
      assert(qubitOperand.getDefiningOp<quir::DeclareQubitOp>() &&
             "could not find the qubit op");
      auto opDeclOp = qubitOperand.getDefiningOp<quir::DeclareQubitOp>();
      auto qubitId = quir::lookupQubitId(opDeclOp);
      if (qubitId.has_value())
        qubits.push_back(qubitId.value());
      else
        opDeclOp->emitError() << "Could not find qubit id.";
    }
  }
  return qubits;
}

mlir::quir::CircuitOp
LoadPulseCalsPass::getCircuitOp(CallCircuitOp callCircuitOp) {
  auto circuitAttr = callCircuitOp->getAttrOfType<FlatSymbolRefAttr>("callee");
  assert(circuitAttr && "Requires a 'callee' symbol reference attribute");

  auto circuitOp = SymbolTable::lookupNearestSymbolFrom<mlir::quir::CircuitOp>(
      callCircuitOp, circuitAttr);
  assert(circuitOp && "matching circuit not found");
  return circuitOp;
}

llvm::StringRef LoadPulseCalsPass::getArgument() const {
  return "load-pulse-cals";
}

llvm::StringRef LoadPulseCalsPass::getDescription() const {
  return "Load the pulse calibrations, and add them to module";
}

llvm::StringRef LoadPulseCalsPass::getName() const {
  return "Load Pulse Calibrations Pass";
}

void LoadPulseCalsPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<PulseDialect>();
}
