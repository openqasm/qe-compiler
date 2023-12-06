//===- LabelQUIRCircuits.cpp - Add attrs for circuit arguments  -*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file implements a pass for adding argument attributes with default
/// values for angle and duration arguments.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/Transforms/QUIRCircuitsAnalysis.h"

#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/QCS/IR/QCSOps.h"
#include "Dialect/QCS/Utils/ParameterInitialValueAnalysis.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Utils/Utils.h"
#include "mlir/Pass/AnalysisManager.h"

#include "mlir/IR/Builders.h"

using namespace mlir;

namespace mlir::quir {

double
parameterValToDouble(mlir::qcs::ParameterLoadOp defOp,
                     mlir::qcs::ParameterInitialValueAnalysis &nameAnalysis) {
  // this method must be called from a pass being
  // managed via a pass manager
  // getCachedParentAnalysis will trigger an assert
  // if the pass is not properly initialized
  return std::get<double>(defOp.getInitialValue(nameAnalysis.getNames()));
}

double
angleValToDouble(mlir::Value inVal,
                 mlir::qcs::ParameterInitialValueAnalysis &nameAnalysis) {
  double retVal = 0.0;
  if (auto defOp = inVal.getDefiningOp<mlir::quir::ConstantOp>()) {
    retVal = defOp.getAngleValueFromConstant().convertToDouble();
  } else if (auto defOp = inVal.getDefiningOp<mlir::qcs::ParameterLoadOp>()) {
    retVal = parameterValToDouble(defOp, nameAnalysis);
  } else if (auto blockArg = inVal.dyn_cast<mlir::BlockArgument>()) {
    auto circuitOp = mlir::dyn_cast<mlir::quir::CircuitOp>(
        inVal.getParentBlock()->getParentOp());
    assert(circuitOp && "can only handle circuit arguments");
    auto argAttr = circuitOp.getArgAttrOfType<mlir::quir::AngleAttr>(
        blockArg.getArgNumber(), mlir::quir::getAngleAttrName());
    retVal = argAttr.getValue().convertToDouble();
  } else if (auto castOp = inVal.getDefiningOp<mlir::oq3::CastOp>()) {
    auto defOp = castOp.arg().getDefiningOp<mlir::qcs::ParameterLoadOp>();
    if (defOp) {
      retVal = parameterValToDouble(defOp, nameAnalysis);
    } else if (auto constOp =
                   castOp.arg().getDefiningOp<mlir::arith::ConstantOp>()) {
      if (auto angleAttr = constOp.getValue().dyn_cast<mlir::quir::AngleAttr>())
        retVal = angleAttr.getValue().convertToDouble();
      else if (auto floatAttr = constOp.getValue().dyn_cast<mlir::FloatAttr>())
        retVal = floatAttr.getValue().convertToDouble();
      else
        inVal.getDefiningOp()->emitOpError()
            << "unable to cast Angle from constant op\n";
    } else {
      inVal.getDefiningOp()->emitOpError()
          << "unable to cast Angle from defining op\n";
    }
  } else {
    inVal.getDefiningOp()->emitOpError()
        << "Non-constant angles are not supported!\n";
  }
  return retVal;
} // angleValToDouble

QUIRCircuitsAnalysis::QUIRCircuitsAnalysis(mlir::Operation *moduleOp,
                                           AnalysisManager &am) {

  if (not invalid_)
    return;

  bool runGetAnalysis = true;
  // auto nameAnalysis =
  //     am.getAnalysis<mlir::qcs::ParameterInitialValueAnalysis>();
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

    // add angle value attributes to all the angle arguments
    for (uint ii = 0; ii < callCircuitOp.operands().size(); ++ii) {

      double value = 0;
      llvm::StringRef parameterName = {};
      quir::DurationAttr duration;

      // track angle values
      if (auto angType = callCircuitOp.operands()[ii]
                             .getType()
                             .dyn_cast<quir::AngleType>()) {

        // auto bits = angType.getWidth();
        auto operand = callCircuitOp.operands()[ii];

        value = angleValToDouble(operand, *nameAnalysis);

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
        // if (value > 0) {
        //   llvm::errs() << "Caching angle: " <<
        //   circuitOp->getParentOfType<ModuleOp>().sym_name() << " " <<
        //   circuitOp.sym_name() << " " << ii << " "; llvm::errs() << value <<
        //   " parameterName " << parameterName << "\n";
        // }
        circuitOperands[circuitOp->getParentOfType<ModuleOp>()][circuitOp][ii] =
            {value, parameterName, duration};
      }

      // track durations
      if (auto durType = callCircuitOp.operands()[ii]
                             .getType()
                             .dyn_cast<quir::DurationType>()) {
        auto operand = dyn_cast<quir::ConstantOp>(
            callCircuitOp.operands()[ii].getDefiningOp());
        operand.dump();
        duration = operand.value().dyn_cast<DurationAttr>();
        duration.dump();
        // llvm::errs() << "Caching duration: " <<
        // circuitOp->getParentOfType<ModuleOp>().sym_name() << " " <<
        // circuitOp.sym_name() << " " << ii << " "; duration.dump();
        circuitOperands[circuitOp->getParentOfType<ModuleOp>()][circuitOp][ii] =
            {value, parameterName, duration};
      }
    }
  });
  invalid_ = false;
}

void QUIRCircuitsAnalysisPass::runOnOperation() {
  mlir::Pass::getAnalysis<QUIRCircuitsAnalysis>();
} // ParameterInitialValueAnalysisPass::runOnOperation()

llvm::StringRef QUIRCircuitsAnalysisPass::getArgument() const {
  return "quir-circuit-analysis";
}

llvm::StringRef QUIRCircuitsAnalysisPass::getDescription() const {
  return "Analyze Circuit Inputs";
}

} // namespace mlir::quir
