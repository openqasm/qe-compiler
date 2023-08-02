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

namespace mlir::pulse {

struct LoadPulseCalsPass
    : public PassWrapper<LoadPulseCalsPass, OperationPass<ModuleOp>> {
  std::string DEFAULT_PULSE_CALS = "";
  std::string ADDITIONAL_PULSE_CALS = "";

  // this pass receives the path to default pulse calibrations file as input.
  // optionally, it can also receive a path to additional pulse calibrations,
  // which can be used to (a) override the pulse calibration that will be used
  // for some quantum gates. e.g., one might be interested to study the impact
  // of changing the pulse sequence corresponding to cx quantum gate on qubits
  // 4 and 5, then they can specify the desired pulse sequence in an additional
  // file; and/or (b) add additional pulse calibrations
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

  // parse the pulse cals and add them to pulseCalsNameToSequenceMap
  void parsePulseCalsSequenceOps(std::string &pulseCalsPath);
  std::map<std::string, SequenceOp> pulseCalsNameToSequenceMap;

  mlir::pulse::SequenceOp
  mergePulseSequenceOps(std::vector<mlir::pulse::SequenceOp> &sequenceOps,
                        const std::string &mergedSequenceOpName);
  bool mergeAttributes(std::vector<mlir::pulse::SequenceOp> &sequenceOps,
                       const std::string &attrName,
                       std::vector<mlir::Attribute> &attrVector);
  std::string getMangledName(std::string &gateName, std::set<uint32_t> &qubits);
  std::string getMangledName(std::string &gateName, uint32_t qubit);
  std::set<uint32_t> getQubitOperands(const std::vector<Value> &qubitOperands,
                                      mlir::quir::CallCircuitOp callCircuitOp);

  // TODO: move this function to Utils; it's used here and MergeCircuitsPass
  static mlir::quir::CircuitOp
  getCircuitOp(mlir::quir::CallCircuitOp callCircuitOp);
};
} // namespace mlir::pulse

#endif // LOAD_PULSE_CALS_H
