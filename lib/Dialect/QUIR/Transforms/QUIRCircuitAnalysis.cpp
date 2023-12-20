//===- QUIRCircuitsAnalsysis.cpp - Cache values for circuits ----*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file implements a analysis for caching argument values with default
/// values for angle and duration arguments.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/QUIRCircuitAnalysis.h"

#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/QCS/IR/QCSOps.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Utils/Utils.h"
#include "mlir/Pass/AnalysisManager.h"

#include "mlir/IR/Builders.h"

using namespace mlir;

namespace mlir::quir {

double
parameterValToDouble(mlir::qcs::ParameterLoadOp defOp,
                     mlir::qcs::ParameterInitialValueAnalysis *nameAnalysis) {
  assert(nameAnalysis &&
         "A valid ParameterInitialValueAnalysis pointer is required");
  return std::get<double>(defOp.getInitialValue(nameAnalysis->getNames()));
}

llvm::Expected<double>
angleValToDouble(mlir::Value inVal,
                 mlir::qcs::ParameterInitialValueAnalysis *nameAnalysis,
                 mlir::quir::QUIRCircuitAnalysis *circuitAnalysis) {

  llvm::StringRef errorStr;

  if (auto defOp = inVal.getDefiningOp<mlir::quir::ConstantOp>())
    return defOp.getAngleValueFromConstant().convertToDouble();

  if (auto defOp = inVal.getDefiningOp<mlir::qcs::ParameterLoadOp>())
    return parameterValToDouble(defOp, nameAnalysis);

  if (auto blockArg = inVal.dyn_cast<mlir::BlockArgument>()) {
    auto circuitOp = mlir::dyn_cast<mlir::quir::CircuitOp>(
        inVal.getParentBlock()->getParentOp());
    assert(circuitOp && "can only handle circuit arguments");

    auto argNum = blockArg.getArgNumber();
    if (circuitAnalysis == nullptr) {

      auto argAttr = circuitOp.getArgAttrOfType<mlir::quir::AngleAttr>(
          argNum, mlir::quir::getAngleAttrName());
      return argAttr.getValue().convertToDouble();
    }
    auto parentModuleOp = circuitOp->getParentOfType<mlir::ModuleOp>();
    return std::get<QUIRCircuitAnalysisEntry::ANGLE>(
        circuitAnalysis->getAnalysisMap()[parentModuleOp][circuitOp][argNum]);
  }

  if (auto castOp = inVal.getDefiningOp<mlir::oq3::CastOp>()) {
    auto defOp = castOp.arg().getDefiningOp<mlir::qcs::ParameterLoadOp>();
    if (defOp)
      return parameterValToDouble(defOp, nameAnalysis);
    if (auto constOp = castOp.arg().getDefiningOp<mlir::arith::ConstantOp>()) {
      if (auto angleAttr = constOp.getValue().dyn_cast<mlir::quir::AngleAttr>())
        return angleAttr.getValue().convertToDouble();
      if (auto floatAttr = constOp.getValue().dyn_cast<mlir::FloatAttr>())
        return floatAttr.getValue().convertToDouble();
      errorStr = "unable to cast Angle from constant op";
    } else {
      errorStr = "unable to cast Angle from defining op";
    }
  } else {
    errorStr = "Non-constant angles are not supported!";
  }
  return llvm::createStringError(llvm::inconvertibleErrorCode(), errorStr);
} // angleValToDouble

double QUIRCircuitAnalysis::getAngleValue(
    mlir::Value operand,
    mlir::qcs::ParameterInitialValueAnalysis *nameAnalysis) {
  assert(nameAnalysis && "valid nameAnalysis pointer required");
  auto valueOrError = angleValToDouble(operand, nameAnalysis);
  if (auto err = valueOrError.takeError()) {
    operand.getDefiningOp()->emitOpError() << toString(std::move(err)) + "\n";
    assert(false && "unhandled value in angleValToDouble");
  }
  return *valueOrError;
}

llvm::StringRef QUIRCircuitAnalysis::getParameterName(mlir::Value operand) {
  llvm::StringRef parameterName = {};
  qcs::ParameterLoadOp parameterLoad;
  parameterLoad = dyn_cast<qcs::ParameterLoadOp>(operand.getDefiningOp());

  if (!parameterLoad) {
    auto castOp = dyn_cast<mlir::oq3::CastOp>(operand.getDefiningOp());
    if (castOp)
      parameterLoad =
          dyn_cast<qcs::ParameterLoadOp>(castOp.arg().getDefiningOp());
  }

  if (parameterLoad &&
      parameterLoad->hasAttr(mlir::quir::getInputParameterAttrName())) {
    parameterName = parameterLoad->getAttrOfType<StringAttr>(
        mlir::quir::getInputParameterAttrName());
  }
  return parameterName;
}

quir::DurationAttr QUIRCircuitAnalysis::getDuration(mlir::Value operand) {
  quir::DurationAttr duration;
  auto constantOp = dyn_cast<quir::ConstantOp>(operand.getDefiningOp());

  if (constantOp)
    return constantOp.value().dyn_cast<DurationAttr>();
  return duration;
}

QUIRCircuitAnalysis::QUIRCircuitAnalysis(mlir::Operation *moduleOp,
                                         AnalysisManager &am) {

  if (not invalid_)
    return;

  bool runGetAnalysis = true;

  mlir::qcs::ParameterInitialValueAnalysis *nameAnalysis;
  auto topLevelModuleOp = moduleOp->getParentOfType<ModuleOp>();
  if (topLevelModuleOp) {
    auto nameAnalysisOptional =
        am.getCachedParentAnalysis<mlir::qcs::ParameterInitialValueAnalysis>(
            moduleOp->getParentOfType<ModuleOp>());
    if (nameAnalysisOptional.hasValue()) {
      nameAnalysis = &nameAnalysisOptional.getValue().get();
      runGetAnalysis = false;
    }
  }

  if (runGetAnalysis)
    nameAnalysis = &am.getAnalysis<mlir::qcs::ParameterInitialValueAnalysis>();

  std::unordered_map<mlir::Operation *, std::map<llvm::StringRef, Operation *>>
      circuitOps;

  moduleOp->walk([&](CircuitOp circuitOp) {
    circuitOps[circuitOp->getParentOfType<ModuleOp>()][circuitOp.sym_name()] =
        circuitOp.getOperation();
  });

  moduleOp->walk([&](CallCircuitOp callCircuitOp) {
    auto search = circuitOps[callCircuitOp->getParentOfType<ModuleOp>()].find(
        callCircuitOp.calleeAttr().getValue());

    if (search ==
        circuitOps[callCircuitOp->getParentOfType<ModuleOp>()].end()) {
      callCircuitOp->emitOpError("Could not find circuit.");
      return;
    }

    auto circuitOp = dyn_cast<CircuitOp>(search->second);
    auto parentModuleOp = circuitOp->getParentOfType<ModuleOp>();

    for (uint ii = 0; ii < callCircuitOp.operands().size(); ++ii) {

      double value = 0;
      llvm::StringRef parameterName = {};
      quir::DurationAttr duration;

      auto operand = callCircuitOp.operands()[ii];

      // cache angle values and parameter names
      if (auto angType = operand.getType().dyn_cast<quir::AngleType>()) {

        value = getAngleValue(operand, nameAnalysis);
        parameterName = getParameterName(operand);
        circuitOperands[parentModuleOp][circuitOp][ii] = {value, parameterName,
                                                          duration};
      }

      // cache durations
      if (auto durType = operand.getType().dyn_cast<quir::DurationType>()) {

        duration = getDuration(operand);
        circuitOperands[parentModuleOp][circuitOp][ii] = {value, parameterName,
                                                          duration};
      }
    }
  });
  invalid_ = false;
}

void QUIRCircuitAnalysisPass::runOnOperation() {
  mlir::Pass::getAnalysis<QUIRCircuitAnalysis>();
} // ParameterInitialValueAnalysisPass::runOnOperation()

llvm::StringRef QUIRCircuitAnalysisPass::getArgument() const {
  return "quir-circuit-analysis";
}

llvm::StringRef QUIRCircuitAnalysisPass::getDescription() const {
  return "Analyze Circuit Inputs";
}

llvm::StringRef QUIRCircuitAnalysisPass::getName() const { return passName; }

} // namespace mlir::quir
