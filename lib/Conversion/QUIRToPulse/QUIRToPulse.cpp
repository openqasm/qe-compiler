//===- QUIRToPulse.cpp - Convert QUIR to Pulse Dialect ----------*- C++ -*-===//
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
///  This file implements the pass for converting QUIR circuits to Pulse
///  sequences
///
//===----------------------------------------------------------------------===//

#include "Conversion/QUIRToPulse/QUIRToPulse.h"

#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/SourceMgr.h"

#define DEBUG_TYPE "QUIRToPulseDebug"

using namespace mlir;
using namespace mlir::quir;
using namespace mlir::oq3;
using namespace mlir::pulse;

void QUIRToPulsePass::runOnOperation() {

  // check for command line override of the path to waveform container
  if (waveformContainer.hasValue())
    WAVEFORM_CONTAINER = waveformContainer.getValue();

  // parse the waveform container ops
  if (!WAVEFORM_CONTAINER.empty())
    parsePulseWaveformContainerOps(WAVEFORM_CONTAINER);

  ModuleOp moduleOp = getOperation();
  FuncOp mainFunc = dyn_cast<FuncOp>(quir::getMainFunction(moduleOp));
  assert(mainFunc && "could not find the main func");

  mainFuncFirstOp = &mainFunc.body().front().front();

  // convert all QUIR circuits to Pulse sequences
  moduleOp->walk([&](CallCircuitOp callCircOp) {
    convertCircuitToSequence(callCircOp, mainFunc, moduleOp);
  });

  // first erase the quir call circuits
  LLVM_DEBUG(llvm::dbgs() << "\nErasing quir call circuits:\n");
  for (auto *op : quirCallCircuitEraseList) {
    LLVM_DEBUG(op->dump());
    op->erase();
  }

  // erase the quir circuits
  LLVM_DEBUG(llvm::dbgs() << "\nErasing quir circuits:\n");
  for (auto *op : quirCircuitEraseList) {
    LLVM_DEBUG(op->dump());
    op->erase();
  }

  // erase the quir circuit operands
  LLVM_DEBUG(llvm::dbgs() << "\nErasing quir circuit operands:\n");
  for (auto *op : quirCircuitOperandEraseList) {
    LLVM_DEBUG(op->dump());
    op->erase();
  }
}

void QUIRToPulsePass::convertCircuitToSequence(CallCircuitOp callCircuitOp,
                                               FuncOp &mainFunc,
                                               ModuleOp moduleOp) {
  mlir::OpBuilder builder(mainFunc);

  auto circuitOp = getCircuitOp(callCircuitOp);
  std::string circName = circuitOp.sym_name().str();
  LLVM_DEBUG(llvm::dbgs() << "\nConverting QUIR circuit " << circName << ":\n");
  assert(callCircuitOp && "callCircuit op is null");
  assert(circuitOp && "circuit op is null");
  addCallCircuitToEraseList(callCircuitOp);
  addCircuitToEraseList(circuitOp);

  // build an empty pulse sequence
  SmallVector<Value> arguments;
  auto argumentsValueRange = ValueRange(arguments.data(), arguments.size());
  auto funcType = builder.getFunctionType(TypeRange(argumentsValueRange), {});
  llvm::SmallVector<mlir::Type> convertedPulseSequenceOpReturnTypes;
  llvm::SmallVector<mlir::Value> convertedPulseSequenceOpReturnValues;
  auto convertedPulseSequenceOp = builder.create<mlir::pulse::SequenceOp>(
      circuitOp.getLoc(), StringRef(circName + "_sequence"), funcType,
      mlir::StringAttr{});
  auto *entryBlock = convertedPulseSequenceOp.addEntryBlock();
  auto entryBuilder = builder.atBlockBegin(entryBlock);

  // reset the circuit conversion helper data structures
  convertedSequenceOpArgIndex = 0;
  circuitArgToConvertedSequenceArgMap.clear();
  convertedPulseSequenceOpArgs.clear();
  convertedPulseCallSequenceOpOperandNames.clear();

  // convert quir circuit args if not already converted, and add the converted
  // args to the the converted pulse sequence
  LLVM_DEBUG(llvm::dbgs() << "Processing QUIR circuit args.\n");
  processCircuitArgs(callCircuitOp, circuitOp, convertedPulseSequenceOp,
                     mainFunc, entryBuilder);

  circuitOp->walk([&](Operation *quirOp) {
    if (quirOp->hasAttr("pulse.calName")) {
      std::string pulseCalName =
          quirOp->getAttrOfType<StringAttr>("pulse.calName").getValue().str();
      SmallVector<Value> pulseCalSequenceArgs;
      Operation *findOp = SymbolTable::lookupSymbolIn(moduleOp, pulseCalName);
      auto pulseCalSequenceOp = dyn_cast<SequenceOp>(findOp);

      LLVM_DEBUG(llvm::dbgs() << "Processing Pulse cal args.\n");
      LLVM_DEBUG(llvm::dbgs() << "QUIR op: ");
      LLVM_DEBUG(quirOp->dump());
      LLVM_DEBUG(llvm::dbgs() << "QUIR op Pulse cal: ");
      LLVM_DEBUG(pulseCalSequenceOp->dump());
      processPulseCalArgs(quirOp, pulseCalSequenceOp, pulseCalSequenceArgs,
                          convertedPulseSequenceOp, mainFunc, entryBuilder);

      auto pulseCalCallSequenceOp =
          entryBuilder.create<mlir::pulse::CallSequenceOp>(
              quirOp->getLoc(), pulseCalSequenceOp, pulseCalSequenceArgs);
      pulseCalCallSequenceOp->setAttr(
          "pulse.operands",
          pulseCalSequenceOp->getAttrOfType<ArrayAttr>("pulse.args"));
      for (auto type : pulseCalCallSequenceOp.getResultTypes())
        convertedPulseSequenceOpReturnTypes.push_back(type);
      for (auto val : pulseCalCallSequenceOp.res())
        convertedPulseSequenceOpReturnValues.push_back(val);

      // add starting timepoint for delayOp
      if (auto delayOp = dyn_cast<mlir::quir::DelayOp>(quirOp)) {
        uint64_t durValue = 0;
        if (delayOp.time().isa<BlockArgument>()) {
          uint argNum = delayOp.time().dyn_cast<BlockArgument>().getArgNumber();
          auto durOpConstantOp = callCircuitOp.getOperand(argNum)
                                     .getDefiningOp<mlir::quir::ConstantOp>();
          auto durOp = quir::getDuration(durOpConstantOp).get();
          durValue = static_cast<uint>(durOp.getDuration().convertToDouble());
          assert(durOp.getType().dyn_cast<DurationType>().getUnits() ==
                     TimeUnits::dt &&
                 "this pass only accepts durations with dt unit");
        } else {
          auto durOp = quir::getDuration(delayOp).get();
          durValue = static_cast<uint>(durOp.getDuration().convertToDouble());
          assert(durOp.getType().dyn_cast<DurationType>().getUnits() ==
                     TimeUnits::dt &&
                 "this pass only accepts durations with dt unit");
        }
        PulseOpSchedulingInterface::setDuration(pulseCalCallSequenceOp,
                                                durValue);
      }
    } else
      assert(((isa<quir::ConstantOp>(quirOp) or isa<quir::ReturnOp>(quirOp) or
               isa<quir::CircuitOp>(quirOp))) &&
             "quir op is not allowed in this pass.");
  });

  // update the converted pulse sequence func and add a return op
  auto newFuncType = builder.getFunctionType(
      TypeRange(convertedPulseSequenceOp.getArguments()),
      TypeRange(convertedPulseSequenceOpReturnTypes.data(),
                convertedPulseSequenceOpReturnTypes.size()));
  convertedPulseSequenceOp.setType(newFuncType);
  entryBuilder.create<mlir::pulse::ReturnOp>(
      convertedPulseSequenceOp.back().back().getLoc(),
      mlir::ValueRange{convertedPulseSequenceOpReturnValues});
  convertedPulseSequenceOp->moveBefore(mainFunc);

  // create a call sequence op for the converted pulse sequence
  auto convertedPulseCallSequenceOp =
      builder.create<mlir::pulse::CallSequenceOp>(callCircuitOp->getLoc(),
                                                  convertedPulseSequenceOp,
                                                  convertedPulseSequenceOpArgs);
  convertedPulseCallSequenceOp->moveAfter(callCircuitOp);
  convertedPulseSequenceOp->setAttr(
      "pulse.args",
      builder.getArrayAttr(convertedPulseCallSequenceOpOperandNames));
  convertedPulseCallSequenceOp->setAttr(
      "pulse.operands",
      builder.getArrayAttr(convertedPulseCallSequenceOpOperandNames));
}

void QUIRToPulsePass::processCircuitArgs(
    mlir::quir::CallCircuitOp callCircuitOp, mlir::quir::CircuitOp circuitOp,
    SequenceOp convertedPulseSequenceOp, FuncOp &mainFunc,
    mlir::OpBuilder &builder) {
  for (uint cnt = 0; cnt < circuitOp.getNumArguments(); cnt++) {
    auto arg = circuitOp.getArgument(cnt);
    auto dictArg = circuitOp.getArgAttrDict(cnt);
    mlir::Type argumentType = arg.getType();
    if (argumentType.isa<mlir::quir::AngleType>()) {
      auto *angleOp = callCircuitOp.getOperand(cnt).getDefiningOp();
      LLVM_DEBUG(llvm::dbgs() << "angle argument ");
      LLVM_DEBUG(angleOp->dump());
      convertedPulseSequenceOp.insertArgument(convertedSequenceOpArgIndex,
                                              builder.getF64Type(), dictArg,
                                              arg.getLoc());
      circuitArgToConvertedSequenceArgMap[cnt] = convertedSequenceOpArgIndex;
      auto convertedAngleToF64 = convertAngleToF64(angleOp, builder);
      convertedSequenceOpArgIndex += 1;
      convertedPulseCallSequenceOpOperandNames.push_back(
          builder.getStringAttr("angle"));
      convertedPulseSequenceOpArgs.push_back(convertedAngleToF64);
    } else if (argumentType.isa<mlir::quir::DurationType>()) {
      auto *durationOp = callCircuitOp.getOperand(cnt).getDefiningOp();
      LLVM_DEBUG(llvm::dbgs() << "duration argument ");
      LLVM_DEBUG(durationOp->dump());
      convertedPulseSequenceOp.insertArgument(convertedSequenceOpArgIndex,
                                              builder.getI64Type(), dictArg,
                                              arg.getLoc());
      circuitArgToConvertedSequenceArgMap[cnt] = convertedSequenceOpArgIndex;
      auto convertedDurationToI64 = convertDurationToI64(
          callCircuitOp, durationOp, cnt, builder, mainFunc);
      convertedSequenceOpArgIndex += 1;
      convertedPulseCallSequenceOpOperandNames.push_back(
          builder.getStringAttr("duration"));
      convertedPulseSequenceOpArgs.push_back(convertedDurationToI64);
    } else if (argumentType.isa<mlir::quir::QubitType>()) {
      auto *qubitOp = callCircuitOp.getOperand(cnt).getDefiningOp();
      addCircuitOperandToEraseList(qubitOp);
    }

    else
      llvm_unreachable("unkown circuit argument.");
  }
}

void QUIRToPulsePass::processPulseCalArgs(
    mlir::Operation *quirOp, SequenceOp pulseCalSequenceOp,
    SmallVector<Value> &pulseCalSequenceArgs,
    SequenceOp convertedPulseSequenceOp, FuncOp &mainFunc,
    mlir::OpBuilder &builder) {

  // get the classical operands of the quir op
  std::queue<Value> angleOperands;
  std::queue<Value> durationOperands;
  getQUIROpClassicalOperands(quirOp, angleOperands, durationOperands);

  assert(pulseCalSequenceOp->hasAttrOfType<ArrayAttr>("pulse.args") and
         "no pulse.args found for the pulse cal sequence.");
  assert(pulseCalSequenceOp->hasAttrOfType<ArrayAttr>("pulse.argPorts") and
         "no pulse.argPorts found for the pulse cal sequence.");

  auto argAttr = pulseCalSequenceOp->getAttrOfType<ArrayAttr>("pulse.args");
  auto argPortsAttr =
      pulseCalSequenceOp->getAttrOfType<ArrayAttr>("pulse.argPorts");

  for (auto const &argumentResult :
       llvm::enumerate(pulseCalSequenceOp.getArguments())) {
    auto index = argumentResult.index();
    mlir::Type argumentType = argumentResult.value().getType();
    mlir::Value argumentValue = argumentResult.value();
    if (argumentType.isa<WaveformType>()) {
      std::string wfrName =
          argAttr[index].dyn_cast<StringAttr>().getValue().str();
      LLVM_DEBUG(llvm::dbgs() << "waveform argument " << wfrName << "\n");
      processWfrOpArg(wfrName, convertedPulseSequenceOp, pulseCalSequenceArgs,
                      argumentValue, mainFunc, builder);
    } else if (argumentType.isa<MixedFrameType>()) {
      std::string const &mixFrameName =
          argAttr[index].dyn_cast<StringAttr>().getValue().str();
      std::string const &portName =
          argPortsAttr[index].dyn_cast<StringAttr>().getValue().str();
      LLVM_DEBUG(llvm::dbgs() << "mixframe argument " << mixFrameName << "\n");
      processMixFrameOpArg(mixFrameName, portName, convertedPulseSequenceOp,
                           pulseCalSequenceArgs, argumentValue, mainFunc,
                           builder);
    } else if (argumentType.isa<PortType>()) {
      std::string const &portName =
          argPortsAttr[index].dyn_cast<StringAttr>().getValue().str();
      LLVM_DEBUG(llvm::dbgs() << "port argument " << portName << "\n");
      processPortOpArg(portName, convertedPulseSequenceOp, pulseCalSequenceArgs,
                       argumentValue, mainFunc, builder);
    } else if (argumentType.isa<FloatType>()) {
      assert(argAttr[index].dyn_cast<StringAttr>().getValue().str() ==
                 "angle" &&
             "unkown argument.");
      assert(angleOperands.size() && "no angle operand found.");
      auto nextAngle = angleOperands.front();
      LLVM_DEBUG(llvm::dbgs() << "angle argument ");
      LLVM_DEBUG(nextAngle.dump());
      processAngleArg(nextAngle, convertedPulseSequenceOp, pulseCalSequenceArgs,
                      builder);
      angleOperands.pop();
    } else if (argumentType.isa<IntegerType>()) {
      assert(argAttr[index].dyn_cast<StringAttr>().getValue().str() ==
                 "duration" &&
             "unkown argument.");
      assert(durationOperands.size() && "no duration operand found.");
      auto nextDuration = durationOperands.front();
      LLVM_DEBUG(llvm::dbgs() << "duration argument ");
      LLVM_DEBUG(nextDuration.dump());
      processDurationArg(nextDuration, convertedPulseSequenceOp,
                         pulseCalSequenceArgs, builder);
      durationOperands.pop();
    } else
      llvm_unreachable("unkown argument type.");
  }
}

void QUIRToPulsePass::getQUIROpClassicalOperands(
    mlir::Operation *quirOp, std::queue<Value> &angleOperands,
    std::queue<Value> &durationOperands) {

  std::vector<Value> classicalOperands;
  if (auto castOp = dyn_cast<CallGateOp>(quirOp))
    classicalCallOperands(castOp, classicalOperands);
  else if (auto castOp = dyn_cast<mlir::quir::DelayOp>(quirOp))
    classicalOperands.push_back(castOp.time());
  else if (auto castOp = dyn_cast<Builtin_UOp>(quirOp)) {
    classicalOperands.push_back(castOp.theta());
    classicalOperands.push_back(castOp.phi());
    classicalOperands.push_back(castOp.lambda());
  }

  for (auto operand : classicalOperands)
    if (operand.getType().isa<mlir::quir::AngleType>())
      angleOperands.push(operand);
    else if (operand.getType().isa<mlir::quir::DurationType>())
      durationOperands.push(operand);
    else
      llvm_unreachable("unkown operand.");
}

void QUIRToPulsePass::processMixFrameOpArg(
    std::string const &mixFrameName, std::string const &portName,
    SequenceOp convertedPulseSequenceOp,
    SmallVector<Value> &pulseCalSequenceArgs, Value argumentValue,
    FuncOp &mainFunc, mlir::OpBuilder &builder) {
  auto mixedFrameOp =
      addMixFrameOpToIR(mixFrameName, portName, mainFunc, builder);
  auto it = std::find(convertedPulseCallSequenceOpOperandNames.begin(),
                      convertedPulseCallSequenceOpOperandNames.end(),
                      builder.getStringAttr(mixFrameName));
  if (it == convertedPulseCallSequenceOpOperandNames.end()) {
    convertedPulseCallSequenceOpOperandNames.push_back(
        builder.getStringAttr(mixFrameName));
    convertedPulseSequenceOpArgs.push_back(mixedFrameOp);
    convertedPulseSequenceOp.insertArgument(
        convertedSequenceOpArgIndex,
        builder.getType<mlir::pulse::MixedFrameType>(), DictionaryAttr{},
        argumentValue.getLoc());
    pulseCalSequenceArgs.push_back(
        convertedPulseSequenceOp.getArguments()[convertedSequenceOpArgIndex]);
    convertedSequenceOpArgIndex += 1;
  } else {
    uint mixFrameOperandIndex =
        std::distance(convertedPulseCallSequenceOpOperandNames.begin(), it);
    pulseCalSequenceArgs.push_back(
        convertedPulseSequenceOp.getArguments()[mixFrameOperandIndex]);
  }
}

void QUIRToPulsePass::processPortOpArg(std::string const &portName,
                                       SequenceOp convertedPulseSequenceOp,
                                       SmallVector<Value> &pulseCalSequenceArgs,
                                       Value argumentValue, FuncOp &mainFunc,
                                       mlir::OpBuilder &builder) {
  auto portOp = addPortOpToIR(portName, mainFunc, builder);
  auto it = std::find(convertedPulseCallSequenceOpOperandNames.begin(),
                      convertedPulseCallSequenceOpOperandNames.end(),
                      builder.getStringAttr(portName));
  if (it == convertedPulseCallSequenceOpOperandNames.end()) {
    convertedPulseCallSequenceOpOperandNames.push_back(
        builder.getStringAttr(portName));
    convertedPulseSequenceOpArgs.push_back(portOp);
    convertedPulseSequenceOp.insertArgument(
        convertedSequenceOpArgIndex, builder.getType<mlir::pulse::PortType>(),
        DictionaryAttr{}, argumentValue.getLoc());
    pulseCalSequenceArgs.push_back(
        convertedPulseSequenceOp.getArguments()[convertedSequenceOpArgIndex]);
    convertedSequenceOpArgIndex += 1;
  } else {
    uint portOperandIndex =
        std::distance(convertedPulseCallSequenceOpOperandNames.begin(), it);
    pulseCalSequenceArgs.push_back(
        convertedPulseSequenceOp.getArguments()[portOperandIndex]);
  }
}

void QUIRToPulsePass::processWfrOpArg(std::string const &wfrName,
                                      SequenceOp convertedPulseSequenceOp,
                                      SmallVector<Value> &pulseCalSequenceArgs,
                                      Value argumentValue, FuncOp &mainFunc,
                                      mlir::OpBuilder &builder) {
  auto wfrOp = addWfrOpToIR(wfrName, mainFunc, builder);
  auto it = std::find(convertedPulseCallSequenceOpOperandNames.begin(),
                      convertedPulseCallSequenceOpOperandNames.end(),
                      builder.getStringAttr(wfrName));
  if (it == convertedPulseCallSequenceOpOperandNames.end()) {
    convertedPulseCallSequenceOpOperandNames.push_back(
        builder.getStringAttr(wfrName));
    convertedPulseSequenceOpArgs.push_back(wfrOp);
    convertedPulseSequenceOp.insertArgument(
        convertedSequenceOpArgIndex,
        builder.getType<mlir::pulse::WaveformType>(), DictionaryAttr{},
        argumentValue.getLoc());
    pulseCalSequenceArgs.push_back(
        convertedPulseSequenceOp.getArguments()[convertedSequenceOpArgIndex]);
    convertedSequenceOpArgIndex += 1;
  } else {
    uint wfrOperandIndex =
        std::distance(convertedPulseCallSequenceOpOperandNames.begin(), it);
    pulseCalSequenceArgs.push_back(
        convertedPulseSequenceOp.getArguments()[wfrOperandIndex]);
  }
}

void QUIRToPulsePass::processAngleArg(Value nextAngleOperand,
                                      SequenceOp convertedPulseSequenceOp,
                                      SmallVector<Value> &pulseCalSequenceArgs,
                                      mlir::OpBuilder &entryBuilder) {
  if (nextAngleOperand.isa<BlockArgument>()) {
    uint circNum = nextAngleOperand.dyn_cast<BlockArgument>().getArgNumber();
    pulseCalSequenceArgs.push_back(
        convertedPulseSequenceOp
            .getArguments()[circuitArgToConvertedSequenceArgMap[circNum]]);
  } else {
    auto angleOp = nextAngleOperand.getDefiningOp<mlir::quir::ConstantOp>();
    std::string angleLocHash =
        std::to_string(mlir::hash_value(angleOp->getLoc()));
    if (classicalQUIROpLocToConvertedPulseOpMap.find(angleLocHash) ==
        classicalQUIROpLocToConvertedPulseOpMap.end()) {
      double angleVal = angleOp.getAngleValueFromConstant().convertToDouble();
      auto f64Angle = entryBuilder.create<mlir::arith::ConstantOp>(
          angleOp.getLoc(), entryBuilder.getFloatAttr(entryBuilder.getF64Type(),
                                                      llvm::APFloat(angleVal)));
      classicalQUIROpLocToConvertedPulseOpMap[angleLocHash] = f64Angle;
    }
    pulseCalSequenceArgs.push_back(
        classicalQUIROpLocToConvertedPulseOpMap[angleLocHash]);
  }
}

void QUIRToPulsePass::processDurationArg(
    Value nextDurationOperand, SequenceOp convertedPulseSequenceOp,
    SmallVector<Value> &pulseCalSequenceArgs, mlir::OpBuilder &entryBuilder) {
  if (nextDurationOperand.isa<BlockArgument>()) {
    uint circNum = nextDurationOperand.dyn_cast<BlockArgument>().getArgNumber();
    pulseCalSequenceArgs.push_back(
        convertedPulseSequenceOp
            .getArguments()[circuitArgToConvertedSequenceArgMap[circNum]]);
  } else {
    auto durationOp =
        nextDurationOperand.getDefiningOp<mlir::quir::ConstantOp>();
    std::string durLocHash =
        std::to_string(mlir::hash_value(nextDurationOperand.getLoc()));
    auto durVal =
        quir::getDuration(durationOp).get().getDuration().convertToDouble();
    assert(durationOp.getType().dyn_cast<DurationType>().getUnits() ==
               TimeUnits::dt &&
           "this pass only accepts durations with dt unit");

    if (classicalQUIROpLocToConvertedPulseOpMap.find(durLocHash) ==
        classicalQUIROpLocToConvertedPulseOpMap.end()) {
      auto dur64 = entryBuilder.create<mlir::arith::ConstantOp>(
          durationOp.getLoc(),
          entryBuilder.getIntegerAttr(entryBuilder.getI64Type(),
                                      uint64_t(durVal)));
      classicalQUIROpLocToConvertedPulseOpMap[durLocHash] = dur64;
    }
    pulseCalSequenceArgs.push_back(
        classicalQUIROpLocToConvertedPulseOpMap[durLocHash]);
  }
}

mlir::Value QUIRToPulsePass::convertAngleToF64(Operation *angleOp,
                                               mlir::OpBuilder &builder) {
  assert(angleOp && "angle op is null");
  std::string angleLocHash =
      std::to_string(mlir::hash_value(angleOp->getLoc()));
  if (classicalQUIROpLocToConvertedPulseOpMap.find(angleLocHash) ==
      classicalQUIROpLocToConvertedPulseOpMap.end()) {
    if (auto castOp = dyn_cast<quir::ConstantOp>(angleOp)) {
      addCircuitOperandToEraseList(angleOp);
      double angleVal = castOp.getAngleValueFromConstant().convertToDouble();
      auto f64Angle = builder.create<mlir::arith::ConstantOp>(
          castOp->getLoc(),
          builder.getFloatAttr(builder.getF64Type(), llvm::APFloat(angleVal)));
      f64Angle->moveAfter(castOp);
      classicalQUIROpLocToConvertedPulseOpMap[angleLocHash] = f64Angle;
    } else if (auto castOp = dyn_cast<qcs::ParameterLoadOp>(angleOp)) {
      auto angleCastedOp = builder.create<oq3::CastOp>(
          castOp->getLoc(), builder.getF64Type(), castOp.res());
      angleCastedOp->moveAfter(castOp);
      classicalQUIROpLocToConvertedPulseOpMap[angleLocHash] = angleCastedOp;
    } else if (auto castOp = dyn_cast<oq3::CastOp>(angleOp)) {
      addCircuitOperandToEraseList(angleOp);
      auto castOpArg = castOp.arg();
      if (auto paramCastOp =
              dyn_cast<qcs::ParameterLoadOp>(castOpArg.getDefiningOp())) {
        auto angleCastedOp = builder.create<oq3::CastOp>(
            paramCastOp->getLoc(), builder.getF64Type(), paramCastOp.res());
        angleCastedOp->moveAfter(paramCastOp);
        classicalQUIROpLocToConvertedPulseOpMap[angleLocHash] = angleCastedOp;
      } else
        llvm_unreachable("castOp arg unknown");
    } else
      llvm_unreachable("angleOp unknown");
  }
  return classicalQUIROpLocToConvertedPulseOpMap[angleLocHash];
}

mlir::Value QUIRToPulsePass::convertDurationToI64(
    mlir::quir::CallCircuitOp callCircuitOp, Operation *durationOp, uint &cnt,
    mlir::OpBuilder &builder, FuncOp &mainFunc) {
  assert(durationOp && "duration op is null");
  std::string durLocHash =
      std::to_string(mlir::hash_value(durationOp->getLoc()));
  if (classicalQUIROpLocToConvertedPulseOpMap.find(durLocHash) ==
      classicalQUIROpLocToConvertedPulseOpMap.end()) {
    if (auto castOp = dyn_cast<quir::ConstantOp>(durationOp)) {
      addCircuitOperandToEraseList(durationOp);
      auto durVal =
          quir::getDuration(castOp).get().getDuration().convertToDouble();
      assert(castOp.getType().dyn_cast<DurationType>().getUnits() ==
                 TimeUnits::dt &&
             "this pass only accepts durations with dt unit");

      auto I64Dur = builder.create<mlir::arith::ConstantOp>(
          castOp->getLoc(),
          builder.getIntegerAttr(builder.getI64Type(), uint64_t(durVal)));
      I64Dur->moveAfter(castOp);
      classicalQUIROpLocToConvertedPulseOpMap[durLocHash] = I64Dur;
    } else
      llvm_unreachable("unkown duration op");
  }
  return classicalQUIROpLocToConvertedPulseOpMap[durLocHash];
}

mlir::pulse::Port_CreateOp
QUIRToPulsePass::addPortOpToIR(std::string const &portName, FuncOp &mainFunc,
                               mlir::OpBuilder &builder) {
  if (openedPorts.find(portName) == openedPorts.end()) {
    auto portOp =
        builder.create<Port_CreateOp>(mainFuncFirstOp->getLoc(), portName);
    portOp->moveBefore(mainFuncFirstOp);
    openedPorts[portName] = portOp;
  }
  return openedPorts[portName];
}

mlir::pulse::MixFrameOp
QUIRToPulsePass::addMixFrameOpToIR(std::string const &mixFrameName,
                                   std::string const &portName,
                                   FuncOp &mainFunc, mlir::OpBuilder &builder) {
  if (openedMixFrames.find(mixFrameName) == openedMixFrames.end()) {
    auto portOp = addPortOpToIR(portName, mainFunc, builder);
    auto mixedFrameOp = builder.create<MixFrameOp>(
        portOp->getLoc(), portOp, builder.getStringAttr(mixFrameName),
        mlir::Value{}, mlir::Value{}, mlir::Value{});
    mixedFrameOp->moveBefore(mainFuncFirstOp);
    openedMixFrames[mixFrameName] = mixedFrameOp;
  }
  return openedMixFrames[mixFrameName];
}

mlir::pulse::Waveform_CreateOp
QUIRToPulsePass::addWfrOpToIR(std::string const &wfrName, FuncOp &mainFunc,
                              mlir::OpBuilder &builder) {
  if (openedWfrs.find(wfrName) == openedWfrs.end()) {
    auto *clonedOp = builder.clone(*pulseNameToWaveformMap[wfrName]);
    auto wfrOp = dyn_cast<Waveform_CreateOp>(clonedOp);
    wfrOp->moveBefore(mainFuncFirstOp);
    openedWfrs[wfrName] = wfrOp;
  }
  return openedWfrs[wfrName];
}

void QUIRToPulsePass::addCircuitToEraseList(mlir::Operation *op) {
  assert(op && "caller requested adding a null op to erase list");
  if (std::find(quirCircuitEraseList.begin(), quirCircuitEraseList.end(), op) ==
      quirCircuitEraseList.end())
    quirCircuitEraseList.push_back(op);
}

void QUIRToPulsePass::addCallCircuitToEraseList(mlir::Operation *op) {
  assert(op && "caller requested adding a null op to erase list");
  if (std::find(quirCallCircuitEraseList.begin(),
                quirCallCircuitEraseList.end(),
                op) == quirCallCircuitEraseList.end())
    quirCallCircuitEraseList.push_back(op);
}

void QUIRToPulsePass::addCircuitOperandToEraseList(mlir::Operation *op) {
  assert(op && "caller requested adding a null op to erase list");
  if (std::find(quirCircuitOperandEraseList.begin(),
                quirCircuitOperandEraseList.end(),
                op) == quirCircuitOperandEraseList.end())
    quirCircuitOperandEraseList.push_back(op);
}

void QUIRToPulsePass::parsePulseWaveformContainerOps(
    std::string &waveformContainerPath) {
  std::string errorMessage;
  llvm::SourceMgr sourceMgr;
  std::unique_ptr<llvm::MemoryBuffer> waveformContainerFile =
      mlir::openInputFile(waveformContainerPath, &errorMessage);
  sourceMgr.AddNewSourceBuffer(std::move(waveformContainerFile), llvm::SMLoc());
  mlir::OwningOpRef<ModuleOp> waveformContainerFileModule(
      mlir::parseSourceFile(sourceMgr, &getContext()));
  assert(waveformContainerFileModule and
         "problem parsing waveform container file");
  auto waveformContainerFileModuleRelease =
      waveformContainerFileModule.release();
  waveformContainerFileModuleRelease->walk([&](mlir::pulse::Waveform_CreateOp
                                                   wfrOp) {
    auto wfrName =
        wfrOp->getAttrOfType<StringAttr>("pulse.waveformName").getValue().str();
    pulseNameToWaveformMap[wfrName] = wfrOp;
  });
}

mlir::quir::CircuitOp
QUIRToPulsePass::getCircuitOp(CallCircuitOp callCircuitOp) {
  auto circuitAttr = callCircuitOp->getAttrOfType<FlatSymbolRefAttr>("callee");
  assert(circuitAttr && "Requires a 'callee' symbol reference attribute");

  auto circuitOp = SymbolTable::lookupNearestSymbolFrom<mlir::quir::CircuitOp>(
      callCircuitOp, circuitAttr);
  assert(circuitOp && "matching circuit not found");
  return circuitOp;
}

llvm::StringRef QUIRToPulsePass::getArgument() const { return "quir-to-pulse"; }

llvm::StringRef QUIRToPulsePass::getDescription() const {
  return "Convert quir circuit to pulse sequence.";
}
