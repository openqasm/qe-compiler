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

#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/SourceMgr.h"

using namespace mlir;
using namespace mlir::quir;
using namespace mlir::pulse;

void LoadPulseCalsPass::runOnOperation() {

  // check for command line override of the path to default pulse cals
  if (defaultPulseCals.hasValue())
    DEFAULT_PULSE_CALS = defaultPulseCals.getValue();

  // parse the default pulse calibrations
  parsePulseCalsSequenceOps(DEFAULT_PULSE_CALS);

  // parse the additional pulse calibrations
  if (!ADDITIONAL_PULSE_CALS.empty())
    parsePulseCalsSequenceOps(ADDITIONAL_PULSE_CALS);

  ModuleOp moduleOp = getOperation();
  FuncOp mainFunc = dyn_cast<FuncOp>(quir::getMainFunction(moduleOp));
  if (!mainFunc)
    assert(false && "could not find the main func");

  moduleOp->walk(
      [&](CallCircuitOp callCircOp) { loadPulseCals(callCircOp, mainFunc); });

  // TODO: add pulseWaveformList ops to the module
}

void LoadPulseCalsPass::loadPulseCals(CallCircuitOp callCircuitOp,
                                      FuncOp funcOp) {

  auto circuitOp = getCircuitOp(callCircuitOp);
  circuitOp->walk([&](Operation *op) {
    if (auto castOp = dyn_cast<CallCircuitOp>(op))
      assert(
          false &&
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
  });
}

void LoadPulseCalsPass::loadPulseCals(CallGateOp callGateOp,
                                      CallCircuitOp callCircuitOp,
                                      FuncOp funcOp) {
  std::vector<Value> qubitOperands;
  qubitCallOperands(callGateOp, qubitOperands);
  std::set<uint32_t> qubits = getQubitOperands(qubitOperands, callCircuitOp);
  std::string gateName = callGateOp.calleeAttr().getValue().str();
  std::string gateMangledName = getMangledName(gateName, qubits);
  if (pulseCalsNameToSequenceMap.find(gateMangledName) ==
      pulseCalsNameToSequenceMap.end())
    assert(false && "could not find any pulse calibration for call gate");

  OpBuilder builder(funcOp.body());
  callGateOp->setAttr("pulse.pulseCalName",
                      builder.getStringAttr(gateMangledName));
  addPulseCalToModule(funcOp, pulseCalsNameToSequenceMap[gateMangledName]);
}

void LoadPulseCalsPass::loadPulseCals(BuiltinCXOp CXOp,
                                      CallCircuitOp callCircuitOp,
                                      FuncOp funcOp) {

  std::vector<Value> qubitOperands;
  qubitOperands.push_back(CXOp.control());
  qubitOperands.push_back(CXOp.target());
  std::set<uint32_t> qubits = getQubitOperands(qubitOperands, callCircuitOp);
  std::string gateName = "cx";
  std::string gateMangledName = getMangledName(gateName, qubits);
  if (pulseCalsNameToSequenceMap.find(gateMangledName) ==
      pulseCalsNameToSequenceMap.end())
    assert(false && "could not find any pulse calibration for the CX gate");

  OpBuilder builder(funcOp.body());
  CXOp->setAttr("pulse.pulseCalName", builder.getStringAttr(gateMangledName));
  addPulseCalToModule(funcOp, pulseCalsNameToSequenceMap[gateMangledName]);
}

void LoadPulseCalsPass::loadPulseCals(Builtin_UOp UOp,
                                      CallCircuitOp callCircuitOp,
                                      FuncOp funcOp) {

  std::vector<Value> qubitOperands;
  qubitOperands.push_back(UOp.target());
  std::set<uint32_t> qubits = getQubitOperands(qubitOperands, callCircuitOp);
  std::string gateName = "u3";
  std::string gateMangledName = getMangledName(gateName, qubits);
  if (pulseCalsNameToSequenceMap.find(gateMangledName) ==
      pulseCalsNameToSequenceMap.end())
    assert(false && "could not find any pulse calibration for the U gate");

  OpBuilder builder(funcOp.body());
  UOp->setAttr("pulse.pulseCalName", builder.getStringAttr(gateMangledName));
  addPulseCalToModule(funcOp, pulseCalsNameToSequenceMap[gateMangledName]);
}

void LoadPulseCalsPass::loadPulseCals(MeasureOp measureOp,
                                      CallCircuitOp callCircuitOp,
                                      FuncOp funcOp) {

  OpBuilder builder(funcOp.body());

  std::vector<Value> qubitOperands;
  qubitCallOperands<MeasureOp>(measureOp, qubitOperands);
  std::set<uint32_t> qubits = getQubitOperands(qubitOperands, callCircuitOp);
  std::string gateName = "measure";
  // check if the measurement is marked with quir.midCircuitMeasure
  if (measureOp->hasAttr("quir.midCircuitMeasure"))
    gateName = "mid_circuit_measure";
  std::string gateMangledName = getMangledName(gateName, qubits);
  measureOp->setAttr("pulse.pulseCalName",
                     builder.getStringAttr(gateMangledName));
  if (pulseCalsNameToSequenceMap.find(gateMangledName) !=
      pulseCalsNameToSequenceMap.end()) {
    // found a pulse calibration for the measurement gate
    addPulseCalToModule(funcOp, pulseCalsNameToSequenceMap[gateMangledName]);
  } else {
    // did not find a pulse calibration for the gate
    // check if there exists pulse calibrations for individual qubits, and if
    // yes, merge them and add the merged pulse sequence to the module
    std::vector<SequenceOp> sequenceOps;
    for (auto &qubit : qubits) {
      std::string individualGateMangledName = getMangledName(gateName, qubit);
      if (pulseCalsNameToSequenceMap.find(individualGateMangledName) ==
          pulseCalsNameToSequenceMap.end())
        assert(false &&
               "could not find pulse calibrations for the measurement gate");
      sequenceOps.push_back(
          pulseCalsNameToSequenceMap[individualGateMangledName]);
    }
    SequenceOp mergedPulseSequenceOp =
        mergePulseSequenceOps(sequenceOps, gateMangledName);

    addPulseCalToModule(funcOp, mergedPulseSequenceOp);
  }
}

void LoadPulseCalsPass::loadPulseCals(mlir::quir::BarrierOp barrierOp,
                                      CallCircuitOp callCircuitOp,
                                      FuncOp funcOp) {

  OpBuilder builder(funcOp.body());

  std::vector<Value> qubitOperands;
  qubitCallOperands<mlir::quir::BarrierOp>(barrierOp, qubitOperands);
  std::set<uint32_t> qubits = getQubitOperands(qubitOperands, callCircuitOp);
  std::string gateName = "barrier";
  std::string gateMangledName = getMangledName(gateName, qubits);
  barrierOp->setAttr("pulse.pulseCalName",
                     builder.getStringAttr(gateMangledName));
  if (pulseCalsNameToSequenceMap.find(gateMangledName) !=
      pulseCalsNameToSequenceMap.end()) {
    // found a pulse calibration for the barrier gate
    addPulseCalToModule(funcOp, pulseCalsNameToSequenceMap[gateMangledName]);
  } else {
    // did not find a pulse calibration for the gate
    // check if there exists pulse calibrations for individual qubits, and if
    // yes, merge them and add the merged pulse sequence to the module
    std::vector<SequenceOp> sequenceOps;
    for (auto &qubit : qubits) {
      std::string individualGateMangledName = getMangledName(gateName, qubit);
      if (pulseCalsNameToSequenceMap.find(individualGateMangledName) ==
          pulseCalsNameToSequenceMap.end())
        assert(false &&
               "could not find pulse calibrations for the barrier gate");
      sequenceOps.push_back(
          pulseCalsNameToSequenceMap[individualGateMangledName]);
    }
    SequenceOp mergedPulseSequenceOp =
        mergePulseSequenceOps(sequenceOps, gateMangledName);

    addPulseCalToModule(funcOp, mergedPulseSequenceOp);
  }
}

void LoadPulseCalsPass::loadPulseCals(mlir::quir::DelayOp delayOp,
                                      CallCircuitOp callCircuitOp,
                                      FuncOp funcOp) {

  OpBuilder builder(funcOp.body());

  std::vector<Value> qubitOperands;
  qubitCallOperands<mlir::quir::DelayOp>(delayOp, qubitOperands);
  std::set<uint32_t> qubits = getQubitOperands(qubitOperands, callCircuitOp);
  std::string gateName = "delay";
  std::string gateMangledName = getMangledName(gateName, qubits);
  delayOp->setAttr("pulse.pulseCalName",
                   builder.getStringAttr(gateMangledName));
  if (pulseCalsNameToSequenceMap.find(gateMangledName) !=
      pulseCalsNameToSequenceMap.end()) {
    // found a pulse calibration for the barrier gate
    addPulseCalToModule(funcOp, pulseCalsNameToSequenceMap[gateMangledName]);
  } else {
    // did not find a pulse calibration for the gate
    // check if there exists pulse calibrations for individual qubits, and if
    // yes, merge them and add the merged pulse sequence to the module
    std::vector<SequenceOp> sequenceOps;
    for (auto &qubit : qubits) {
      std::string individualGateMangledName = getMangledName(gateName, qubit);
      if (pulseCalsNameToSequenceMap.find(individualGateMangledName) ==
          pulseCalsNameToSequenceMap.end())
        assert(false && "could not find pulse calibrations for the delay gate");
      sequenceOps.push_back(
          pulseCalsNameToSequenceMap[individualGateMangledName]);
    }
    SequenceOp mergedPulseSequenceOp =
        mergePulseSequenceOps(sequenceOps, gateMangledName);

    addPulseCalToModule(funcOp, mergedPulseSequenceOp);
  }
}

void LoadPulseCalsPass::loadPulseCals(mlir::quir::ResetQubitOp resetOp,
                                      CallCircuitOp callCircuitOp,
                                      FuncOp funcOp) {

  OpBuilder builder(funcOp.body());

  std::vector<Value> qubitOperands;
  qubitCallOperands<mlir::quir::ResetQubitOp>(resetOp, qubitOperands);
  std::set<uint32_t> qubits = getQubitOperands(qubitOperands, callCircuitOp);
  std::string gateName = "reset";
  std::string gateMangledName = getMangledName(gateName, qubits);
  resetOp->setAttr("pulse.pulseCalName",
                   builder.getStringAttr(gateMangledName));
  if (pulseCalsNameToSequenceMap.find(gateMangledName) !=
      pulseCalsNameToSequenceMap.end()) {
    // found a pulse calibration for the gate
    addPulseCalToModule(funcOp, pulseCalsNameToSequenceMap[gateMangledName]);
  } else {
    // did not find a pulse calibration for the gate
    // check if there exists pulse calibrations for individual qubits, and if
    // yes, merge them and add the merged pulse sequence to the module
    std::vector<SequenceOp> sequenceOps;
    for (auto &qubit : qubits) {
      std::string individualGateMangledName = getMangledName(gateName, qubit);
      if (pulseCalsNameToSequenceMap.find(individualGateMangledName) ==
          pulseCalsNameToSequenceMap.end())
        assert(false && "could not find pulse calibrations for the reset gate");
      sequenceOps.push_back(
          pulseCalsNameToSequenceMap[individualGateMangledName]);
    }
    SequenceOp mergedPulseSequenceOp =
        mergePulseSequenceOps(sequenceOps, gateMangledName);

    addPulseCalToModule(funcOp, mergedPulseSequenceOp);
  }
}

void LoadPulseCalsPass::addPulseCalToModule(
    FuncOp funcOp, mlir::pulse::SequenceOp sequenceOp) {
  OpBuilder builder(funcOp.body());
  auto *clonedPulseCalOp = builder.clone(*sequenceOp);
  auto clonedPulseCalSequenceOp = dyn_cast<SequenceOp>(clonedPulseCalOp);
  clonedPulseCalSequenceOp->moveBefore(funcOp);
}

void LoadPulseCalsPass::parsePulseCalsSequenceOps(std::string &pulseCalsPath) {
  std::string errorMessage;
  llvm::SourceMgr sourceMgr;
  std::unique_ptr<llvm::MemoryBuffer> pulseCalsFile =
      mlir::openInputFile(pulseCalsPath, &errorMessage);
  sourceMgr.AddNewSourceBuffer(std::move(pulseCalsFile), llvm::SMLoc());
  mlir::OwningOpRef<ModuleOp> pulseCalsModule(
      mlir::parseSourceFile(sourceMgr, &getContext()));
  if (!pulseCalsModule)
    assert(false and "problem parsing MLIR pulse calibrations file");
  auto pulseCalsModuleRelease = pulseCalsModule.release();
  pulseCalsModuleRelease->walk([&](mlir::pulse::SequenceOp sequenceOp) {
    auto sequenceName = sequenceOp.sym_name().str();
    pulseCalsNameToSequenceMap[sequenceName] = sequenceOp;
  });
}

mlir::pulse::SequenceOp LoadPulseCalsPass::mergePulseSequenceOps(
    std::vector<mlir::pulse::SequenceOp> &sequenceOps,
    const std::string& mergedSequenceOpName) {

  if (sequenceOps.size() == 0)
    assert(false && "sequence op vector is empty; nothing to merge");

  SequenceOp firstSequenceOp = sequenceOps[0];

  OpBuilder builder(firstSequenceOp);

  llvm::SmallVector<Type> inputTypes;
  llvm::SmallVector<Value> inputValues;
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
  // remove pulse duration attribute if exists
  if (mergedSequenceOp->hasAttr("pulse.duration"))
    mergedSequenceOp->removeAttr("pulse.duration");

  // check if ALL the sequence ops has args/argPorts attr, and if yes,
  // merge the attributes and add them to the merged sequence op
  std::vector<mlir::Attribute> pulseSequenceOpArgs;
  std::vector<mlir::Attribute> pulseSequenceOpArgPorts;
  bool allSequenceOpsHasArgsAttr =
      mergeAttributes(sequenceOps, "pulse.args", pulseSequenceOpArgs);
  bool allSequenceOpsHasArgPortsAttr =
      mergeAttributes(sequenceOps, "pulse.argPorts", pulseSequenceOpArgPorts);

  if (allSequenceOpsHasArgsAttr) {
    mlir::ArrayAttr arrayAttr = builder.getArrayAttr(pulseSequenceOpArgs);
    mergedSequenceOp->setAttr("pulse.args", arrayAttr);
  } else if (mergedSequenceOp->hasAttr("pulse.args"))
    mergedSequenceOp->removeAttr("pulse.args");

  if (allSequenceOpsHasArgPortsAttr) {
    mlir::ArrayAttr arrayAttr = builder.getArrayAttr(pulseSequenceOpArgPorts);
    mergedSequenceOp->setAttr("pulse.argPorts", arrayAttr);
  } else if (mergedSequenceOp->hasAttr("pulse.argPorts"))
    mergedSequenceOp->removeAttr("pulse.argPorts");

  // map original arguments for new sequence based on original sequences'
  // argument numbers
  BlockAndValueMapping mapper;
  auto baseArgNum = mergedSequenceOp.getNumArguments();
  for (std::size_t seqNum = 1; seqNum < sequenceOps.size(); seqNum++) {
    for (uint cnt = 0; cnt < sequenceOps[seqNum].getNumArguments(); cnt++) {
      auto arg = sequenceOps[seqNum].getArgument(cnt);
      auto dictArg = sequenceOps[seqNum].getArgAttrDict(cnt);
      mergedSequenceOp.insertArgument(baseArgNum + cnt, arg.getType(), dictArg,
                                      arg.getLoc());
      mapper.map(arg, mergedSequenceOp.getArgument(baseArgNum + cnt));
    }
  }

  // clone the body of the original sequence ops
  for (std::size_t seqNum = 1; seqNum < sequenceOps.size(); seqNum++) {
    builder.setInsertionPointAfter(&mergedSequenceOp.back().back());
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
  auto opType = mergedSequenceOp.getType();
  mergedSequenceOp.setType(builder.getFunctionType(
      /*inputs=*/opType.getInputs(),
      /*results=*/ArrayRef<Type>(outputTypes)));

  return mergedSequenceOp;
}

bool LoadPulseCalsPass::mergeAttributes(
    std::vector<mlir::pulse::SequenceOp> &sequenceOps, const std::string& attrName,
    std::vector<mlir::Attribute> &attrVector) {

  bool allSequenceOpsHasAttr = true;
  for (auto & sequenceOp : sequenceOps) {
    if (sequenceOp->hasAttr(attrName)) {
      auto pulseArgs =
          sequenceOps[seqNum]->getAttrOfType<ArrayAttr>(attrName);
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
                                              std::set<uint32_t> &qubits) {
  std::string gateMangledName = gateName;
  for (int qubit : qubits) {
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

std::set<uint32_t>
LoadPulseCalsPass::getQubitOperands(const std::vector<Value>& qubitOperands,
                                    CallCircuitOp callCircuitOp) {

  std::set<uint32_t> qubits;
  for (auto &qubit : qubitOperands) {
    if (auto declOp = qubit.getDefiningOp<quir::DeclareQubitOp>()) {
      qubits.insert(*quir::lookupQubitId(qubit));
    } else {
      // qubit is a block argument
      auto blockArg = qubit.dyn_cast<BlockArgument>();
      unsigned argIdx = blockArg.getArgNumber();
      auto qubitOp = callCircuitOp->getOperand(argIdx);
      assert(qubitOp.getDefiningOp<quir::DeclareQubitOp>() &&
             "could not find the qubit op");
      qubits.insert(*quir::lookupQubitId(qubitOp));
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
