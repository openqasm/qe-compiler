//===- QUIRToPulse.h - Convert QUIR to Pulse Dialect ------------*- C++ -*-===//
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
///  This file declares the pass for converting QUIR circuits to Pulse sequences
///
//===----------------------------------------------------------------------===//

#ifndef QUIRTOPULSE_CONVERSION_H
#define QUIRTOPULSE_CONVERSION_H

#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/Pulse/IR/PulseOps.h"
#include "Dialect/QCS/IR/QCSOps.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"

#include <queue>

namespace mlir::pulse {

struct QUIRToPulsePass
    : public PassWrapper<QUIRToPulsePass, OperationPass<ModuleOp>> {
  std::string WAVEFORM_CONTAINER = "";

  // this pass can optionally receive a path to a file containing pulse waveform
  // container operations, which will contain pulse waveform operations that
  // will be passed as argument to pulse calibration sequences.
  QUIRToPulsePass() = default;
  QUIRToPulsePass(const QUIRToPulsePass &pass) : PassWrapper(pass) {}
  QUIRToPulsePass(std::string inWfrContainer) {
    WAVEFORM_CONTAINER = std::move(inWfrContainer);
  }

  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  llvm::StringRef getName() const override;

  // optionally, one can override the path to pulse waveform container file with
  // this option; e.g., to write a LIT test one can invoke this pass with
  // --quir-to-pulse=waveform-container=<path-to-waveform-container-file>
  Option<std::string> waveformContainer{
      *this, "waveform-container",
      llvm::cl::desc("an MLIR file containing waveform container operations"),
      llvm::cl::value_desc("filename"), llvm::cl::init("")};

  mlir::Operation *mainFuncFirstOp;

  // convert quir circuit to pulse sequence
  void convertCircuitToSequence(mlir::quir::CallCircuitOp callCircuitOp,
                                mlir::func::FuncOp &mainFunc,
                                ModuleOp moduleOp);
  // helper datastructure for converting quir circuit to pulse sequence; these
  // will be reset every time convertCircuitToSequence is called and will be
  // used by several functions that are called within that function
  uint convertedSequenceOpArgIndex;
  std::map<uint, uint> circuitArgToConvertedSequenceArgMap;
  SmallVector<Value> convertedPulseSequenceOpArgs;
  std::vector<mlir::Attribute> convertedPulseCallSequenceOpOperandNames;

  // process the args of the circuit op, and add corresponding args to the
  // converted pulse sequence op
  void processCircuitArgs(mlir::quir::CallCircuitOp callCircuitOp,
                          mlir::quir::CircuitOp circuitOp,
                          SequenceOp convertedPulseSequenceOp,
                          mlir::func::FuncOp &mainFunc,
                          mlir::OpBuilder &builder);

  // process the args of the pulse cal sequence op corresponding to quirOp
  void processPulseCalArgs(mlir::Operation *quirOp,
                           SequenceOp pulseCalSequenceOp,
                           SmallVector<Value> &pulseCalSeqArgs,
                           SequenceOp convertedPulseSequenceOp,
                           mlir::func::FuncOp &mainFunc,
                           mlir::OpBuilder &builder);
  void getQUIROpClassicalOperands(mlir::Operation *quirOp,
                                  std::queue<Value> &angleOperands,
                                  std::queue<Value> &durationOperands);
  void processMixFrameOpArg(std::string const &mixFrameName,
                            std::string const &portName,
                            SequenceOp convertedPulseSequenceOp,
                            SmallVector<Value> &quirOpPulseCalSeqArgs,
                            Value argumentValue, mlir::func::FuncOp &mainFunc,
                            mlir::OpBuilder &builder);
  void processPortOpArg(std::string const &portName,
                        SequenceOp convertedPulseSequenceOp,
                        SmallVector<Value> &quirOpPulseCalSeqArgs,
                        Value argumentValue, mlir::func::FuncOp &mainFunc,
                        mlir::OpBuilder &builder);
  void processWfrOpArg(std::string const &wfrName,
                       SequenceOp convertedPulseSequenceOp,
                       SmallVector<Value> &quirOpPulseCalSeqArgs,
                       Value argumentValue, mlir::func::FuncOp &mainFunc,
                       mlir::OpBuilder &builder);
  void processAngleArg(Value nextAngleOperand,
                       SequenceOp convertedPulseSequenceOp,
                       SmallVector<Value> &quirOpPulseCalSeqArgs,
                       mlir::OpBuilder &builder);
  void processDurationArg(Value frontDurOperand,
                          SequenceOp convertedPulseSequenceOp,
                          SmallVector<Value> &quirOpPulseCalSeqArgs,
                          mlir::OpBuilder &builder);

  // convert angle to F64
  mlir::Value convertAngleToF64(Operation *angleOp, mlir::OpBuilder &builder);
  // convert duration to I64
  mlir::Value convertDurationToI64(mlir::quir::CallCircuitOp callCircuitOp,
                                   Operation *durOp, uint &cnt,
                                   mlir::OpBuilder &builder,
                                   mlir::func::FuncOp &mainFunc);
  // map of the hashed location of quir angle/duration ops to their converted
  // pulse ops
  std::map<std::string, mlir::Value> classicalQUIROpLocToConvertedPulseOpMap;

  // port name to Port_CreateOp map
  std::map<std::string, mlir::pulse::Port_CreateOp> openedPorts;
  // mixframe name to MixFrameOp map
  std::map<std::string, mlir::pulse::MixFrameOp> openedMixFrames;
  // waveform name to Waveform_CreateOp map
  std::map<std::string, mlir::pulse::Waveform_CreateOp> openedWfrs;
  // add a port to IR if it's not already added and return the Port_CreateOp
  mlir::pulse::Port_CreateOp addPortOpToIR(std::string const &portName,
                                           mlir::func::FuncOp &mainFunc,
                                           mlir::OpBuilder &builder);
  // add a mixframe to IR if it's not already added and return the MixFrameOp
  mlir::pulse::MixFrameOp addMixFrameOpToIR(std::string const &mixFrameName,
                                            std::string const &portName,
                                            mlir::func::FuncOp &mainFunc,
                                            mlir::OpBuilder &builder);
  // add a waveform to IR if it's not already added and return the
  // Waveform_CreateOp
  mlir::pulse::Waveform_CreateOp addWfrOpToIR(std::string const &wfrName,
                                              mlir::func::FuncOp &mainFunc,
                                              mlir::OpBuilder &builder);

  void addCircuitToEraseList(mlir::Operation *op);
  void addCallCircuitToEraseList(mlir::Operation *op);
  void addCircuitOperandToEraseList(mlir::Operation *op);
  std::vector<mlir::Operation *> quirCircuitEraseList;
  std::vector<mlir::Operation *> quirCallCircuitEraseList;
  std::vector<mlir::Operation *> quirCircuitOperandEraseList;

  // parse the waveform containers and add them to pulseNameToWaveformMap
  void parsePulseWaveformContainerOps(std::string &waveformContainerPath);
  std::map<std::string, Waveform_CreateOp> pulseNameToWaveformMap;

  llvm::StringMap<Operation *> symbolMap;
  mlir::quir::CircuitOp getCircuitOp(mlir::quir::CallCircuitOp callCircuitOp);
  mlir::pulse::SequenceOp getSequenceOp(std::string const &symbolName);
};
} // namespace mlir::pulse

#endif // QUIRTOPULSE_CONVERSION_H
