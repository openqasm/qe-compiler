//===- LoadPulseCals.h ------------------------------------------*- C++ -*-===//
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
/// This file declares the pass to load the pulse calibrations.
///
//===----------------------------------------------------------------------===//

#ifndef LOAD_PULSE_CALS_H
#define LOAD_PULSE_CALS_H

#include "Dialect/Pulse/IR/PulseOps.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include <unordered_set>
#include <vector>

namespace mlir::pulse {

struct LoadPulseCalsPass
    : public PassWrapper<LoadPulseCalsPass, OperationPass<ModuleOp>> {
  std::string DEFAULT_PULSE_CALS = "";
  std::string ADDITIONAL_PULSE_CALS = "";

  // this pass uses up to three sources to obtain the pulse calibration
  // sequences. (1) default pulse calibration file if specified.
  // (2) additional pulse calibration file if specified; this will
  // override default pulse calibrations. (3) pulse calibration
  // sequences specified in the MLIR input program by compiler user;
  // this will override both default and additional pulse calibrations.
  LoadPulseCalsPass() = default;
  LoadPulseCalsPass(const LoadPulseCalsPass &pass) : PassWrapper(pass) {}
  LoadPulseCalsPass(std::string inDefaultPulseCals) {
    DEFAULT_PULSE_CALS = std::move(inDefaultPulseCals);
  }
  LoadPulseCalsPass(std::string inDefaultPulseCals,
                    std::string inAdditionalPulseCals) {
    DEFAULT_PULSE_CALS = std::move(inDefaultPulseCals);
    ADDITIONAL_PULSE_CALS = std::move(inAdditionalPulseCals);
  }

  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  llvm::StringRef getName() const override;
  std::string passName = 
      "Load Pulse Cals Pass (" + getArgument().str() + ")";

  // optionally, one can override the path to default pulse calibrations with
  // this option; e.g., to write a LIT test one can invoke this pass with
  // --load-pulse-cals=default-pulse-cals=<path-to-pulse-cals-file>
  Option<std::string> defaultPulseCals{
      *this, "default-pulse-cals",
      llvm::cl::desc("default pulse calibrations MLIR file"),
      llvm::cl::value_desc("filename"), llvm::cl::init("")};

  void loadPulseCals(mlir::quir::CallCircuitOp callCircuitOp, FuncOp funcOp);
  void loadPulseCals(mlir::quir::CallGateOp callGateOp,
                     mlir::quir::CallCircuitOp callCircuitOp, FuncOp funcOp);
  void loadPulseCals(mlir::quir::BuiltinCXOp CXOp,
                     mlir::quir::CallCircuitOp callCircuitOp, FuncOp funcOp);
  void loadPulseCals(mlir::quir::Builtin_UOp UOp,
                     mlir::quir::CallCircuitOp callCircuitOp, FuncOp funcOp);
  void loadPulseCals(mlir::quir::MeasureOp measureOp,
                     mlir::quir::CallCircuitOp callCircuitOp, FuncOp funcOp);
  void loadPulseCals(mlir::quir::BarrierOp barrierOp,
                     mlir::quir::CallCircuitOp callCircuitOp, FuncOp funcOp);
  void loadPulseCals(mlir::quir::DelayOp delayOp,
                     mlir::quir::CallCircuitOp callCircuitOp, FuncOp funcOp);
  void loadPulseCals(mlir::quir::ResetQubitOp resetOp,
                     mlir::quir::CallCircuitOp callCircuitOp, FuncOp funcOp);

  void addPulseCalToModule(FuncOp funcOp, mlir::pulse::SequenceOp sequenceOp);

  // parse the pulse cals and return the parsed module
  llvm::Error parsePulseCalsModuleOp(std::string &pulseCalsPath,
                                     mlir::OwningOpRef<ModuleOp> &owningOpRef);
  mlir::OwningOpRef<ModuleOp> defaultPulseCalsModule;
  mlir::OwningOpRef<ModuleOp> additionalPulseCalsModule;
  std::map<std::string, SequenceOp> pulseCalsNameToSequenceMap;

  mlir::pulse::SequenceOp
  mergePulseSequenceOps(std::vector<mlir::pulse::SequenceOp> &sequenceOps,
                        const std::string &mergedSequenceOpName);
  // remove the redundant delay args after merging multiple delayOp pulse cals
  void removeRedundantDelayArgs(mlir::pulse::SequenceOp sequenceOp,
                                mlir::OpBuilder &builder);

  // set of pulse cals already added to IR
  std::unordered_set<std::string> pulseCalsAddedToIR;

  // returns true if all the sequence ops in the input vector has the same
  // duration
  bool doAllSequenceOpsHaveSameDuration(
      std::vector<mlir::pulse::SequenceOp> &sequenceOps);
  // returns true if all the sequence ops in the input vector has attrName
  // attribute and if yes, merges the attributes
  bool mergeAttributes(std::vector<mlir::pulse::SequenceOp> &sequenceOps,
                       const std::string &attrName,
                       std::vector<mlir::Attribute> &attrVector);

  std::string getMangledName(std::string &gateName,
                             std::vector<uint32_t> &qubits);
  std::string getMangledName(std::string &gateName, uint32_t qubit);
  std::vector<uint32_t>
  getQubitOperands(std::vector<Value> &qubitOperands,
                   mlir::quir::CallCircuitOp callCircuitOp);

  // TODO: move this function to Utils; it's used here and MergeCircuitsPass
  static mlir::quir::CircuitOp
  getCircuitOp(mlir::quir::CallCircuitOp callCircuitOp);
};
} // namespace mlir::pulse

#endif // LOAD_PULSE_CALS_H
