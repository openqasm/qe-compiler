//===- QUIRToPulse.cpp - Pulse dialect -------------------------*- C++ -*-===//
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
//
#include "Conversion/QUIRToPulse/QUIRToPulse.h"

#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/Pulse/IR/PulseOps.h"
#include "Dialect/Pulse/Transforms/SystemCreation.h"
#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Utils/LegacyInputConversion.h"
#include "Utils/SystemDefinition.h"
#include "Utils/SystemNodes.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/FuncConversions.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include <functional>

namespace qu = qssc::utils;

namespace mlir::pulse {

class DeclareQubitConversionPat
    : public OpConversionPattern<quir::DeclareQubitOp> {
  std::reference_wrapper<qu::LegacyInputConversion> setup_;

  auto hasPort(quir::DeclareQubitOp originalOp, const std::string &id) const
      -> Value {
    for (auto op : originalOp->getBlock()->getOps<Port_CreateOp>())
      if (op && id == op.uid())
        return op;
    return {};
  };

public:
  explicit DeclareQubitConversionPat(
      MLIRContext *ctx, TypeConverter &typeConverter,
      std::reference_wrapper<qu::LegacyInputConversion> setup)
      : OpConversionPattern(typeConverter, ctx, 1), setup_(setup) {}
  LogicalResult
  matchAndRewrite(quir::DeclareQubitOp originalOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    const auto loc = originalOp->getLoc();
    auto *const ctx = originalOp->getContext();

    const auto &qubits = setup_.get().findAll<qu::Qubit>();

    std::vector<Value> participants;
    for (const auto &qubit : qubits) {
      if (qubit->id == originalOp.id()) {
        const auto &ports = setup_.get().findAllFrom<qu::Port>(qubit);

        for (const auto &port : ports) {
          const auto &name = port->id();
          Value v = hasPort(originalOp, name);
          if (!v) {
            participants.emplace_back(rewriter.create<Port_CreateOp>(
                loc, PortType::get(ctx), port->id()));
          } else {
            participants.emplace_back(v);
          }
        }
      }
    }

    if (participants.size() < 1)
      llvm::errs() << "Failed to find relative ports for one of the qubits.";

    // PortGroup has been removed. This should be placed with the correct ports
    // when this pass is updated. 
    // auto portGroupOp = rewriter.create<PortGroup_CreateOp>(
    //     loc, PortGroupType::get(ctx), ValueRange{participants});

    // rewriter.replaceOp(originalOp, {portGroupOp.out()});
    return success();
  }
};

class DefCalConversionPat : public OpConversionPattern<quir::CallDefCalGateOp> {
  std::reference_wrapper<qu::LegacyInputConversion> setup_;
  std::reference_wrapper<SymbolTable> symbolTable_;
  auto getQubitsAndGates(quir::CallDefCalGateOp originalOp,
                         OpAdaptor adaptor) const {
    std::vector<std::shared_ptr<qu::Qubit>> participants;
    std::vector<std::shared_ptr<qu::Gate>> potentialGates;
    std::map<std::string, size_t> gateOperandMap;
    const auto qubits = setup_.get().findAll<qu::Qubit>();
    auto operands = adaptor.getOperands();

    if (qubits.size() < 1)
      llvm::errs() << "Failed to get any references to system graph qubits";

    for (size_t i = 0; i < operands.size(); ++i) {
      const auto arg = originalOp.operands()[i];
      if (auto qubitOp = dyn_cast<quir::DeclareQubitOp>(arg.getDefiningOp())) {
        std::shared_ptr<qu::Qubit> qubit;
        for (const auto &q : qubits) {
          if (q->id == qubitOp.id().getValue()) {
            qubit = q;
            participants.emplace_back(qubit);
            break;
          }
        }
        if (!qubit) {
          llvm::errs() << "Could not find a relative qubit within system graph "
                          "for this defcal: "
                       << originalOp;
        }

        const auto gates = setup_.get().findAllFrom<qu::Gate>(qubit);
        if (gates.size() < 1) {
          llvm::errs() << "Failed to get gates within system graph for a qubit "
                          "associated with this defcal: "
                       << originalOp;
        }

        potentialGates.insert(potentialGates.end(), gates.begin(), gates.end());
        for (const auto &g : gates)
          gateOperandMap.emplace(g->uid(), i);
      }
    }

    return std::make_tuple(participants, potentialGates, gateOperandMap);
  }

  auto
  getRelativeOps(quir::CallDefCalGateOp originalOp,
                 const std::vector<std::shared_ptr<qu::Qubit>> &participants,
                 const std::vector<std::shared_ptr<qu::Gate>> &potentialGates,
                 const std::map<std::string, size_t> &gateOperandMap) const {
    std::map<std::string, size_t> OpOperandMap;
    std::vector<std::shared_ptr<qu::Operation>> ops;
    std::string qubitsStr;
    std::vector<int> gateQubitOrder;

    for (const auto &q : participants) {
      gateQubitOrder.emplace_back(q->id);
      qubitsStr += std::to_string(q->id);
    }

    const auto gateType = originalOp.callee();
    for (const auto &gate : potentialGates) {
      if (gate->type() == gateType && gate->qubits() == gateQubitOrder) {
        ops = setup_.get().getEdgesFrom<qu::Operation>(gate);
        std::sort(ops.begin(), ops.end(),
                  [](const std::shared_ptr<qu::Operation> &lhs,
                     const std::shared_ptr<qu::Operation> &rhs) {
                    return lhs->order() < rhs->order();
                  });
        if (ops.size() < 1) {
          llvm::errs()
              << "Failed to get operations from system graph relative to gate: "
              << gate->type() << " - with qubits (in order): " << qubitsStr;
        }
        for (const auto &op : ops)
          OpOperandMap.emplace(op->uid(), gateOperandMap.at(gate->uid()));
        break;
      }
    }
    return std::make_tuple(ops, OpOperandMap);
  }

public:
  explicit DefCalConversionPat(
      MLIRContext *ctx, TypeConverter &typeConverter,
      std::reference_wrapper<qu::LegacyInputConversion> setup,
      SymbolTable &symbolTable)
      : OpConversionPattern(typeConverter, ctx, 1), setup_(setup),
        symbolTable_(symbolTable) {}
  LogicalResult
  matchAndRewrite(quir::CallDefCalGateOp originalOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    const std::string callee = originalOp.callee().str();
    const auto loc = originalOp->getLoc();
    //auto *const ctx = originalOp->getContext();
    auto *const existingOp = symbolTable_.get().lookup(callee);
    auto operands = adaptor.getOperands();

    FuncOp functionOp;

    if (!existingOp) {

      std::string namingHelpers;
      for (auto i : originalOp.operands())
        if (auto j = dyn_cast<quir::DeclareQubitOp>(i.getDefiningOp()))
          namingHelpers += std::to_string(j.id().getValue());

      std::string functionName = callee + namingHelpers;
      while (symbolTable_.get().lookup(functionName))
        functionName += "_";

      auto funcOp = originalOp->getParentOfType<FuncOp>();
      OpBuilder builder(funcOp);
      functionOp = builder.create<FuncOp>(
          originalOp->getLoc(), functionName,
          FunctionType::get(originalOp->getContext(), TypeRange{operands},
                            TypeRange{}));

      builder = builder.atBlockBegin(functionOp.addEntryBlock());

      // get participants and their relative gates
      auto [participants, potentialGates, gateOperandMap] =
          getQubitsAndGates(originalOp, operands);

      // get relative ops
      auto [ops, OpOperandMap] = getRelativeOps(originalOp, participants,
                                                potentialGates, gateOperandMap);

      for (const auto &op : ops) {
        const auto port = setup_.get().getEdgeTarget<qu::Port>(op);
        if (!port)
          llvm::errs() << "Failed to find target port/frame for operation";

        // Port_SelectOp has been removed
        // auto target = builder.create<Port_SelectOp>(
        //     loc, PortType::get(ctx),
        //     functionOp.getArgument(OpOperandMap.at(op->uid())), port->id());

        // if (const auto play = std::dynamic_pointer_cast<qu::PlayOp>(op)) {
        //   processPlayOps(play, loc, ctx, target, builder);
        // } else if (const auto delay =
        //                std::dynamic_pointer_cast<qu::DelayOp>(op)) {

        //   const IntegerType intType = builder.getI32Type();

        //   const Value duration = builder.create<mlir::arith::ConstantOp>(
        //       loc, intType, builder.getIntegerAttr(intType, delay->duration));

        //   builder.create<DelayOp>(loc, duration, target);
        // } else if (const auto fc =
        //                std::dynamic_pointer_cast<qu::FrameChangeOp>(op)) {
        //   const FloatType floatType = builder.getF64Type();

        //   const Value phase = builder.create<mlir::arith::ConstantOp>(
        //       loc, floatType, builder.getFloatAttr(floatType, fc->phase));

        //   builder.create<ShiftPhaseOp>(loc, target, phase);
        // } else if (const auto acquire =
        //                std::dynamic_pointer_cast<qu::AcquisitionOp>(op)) {
        // } else {
        //   llvm::errs() << "Encountered unsupported operation.";
        // }
      }

      builder.create<mlir::ReturnOp>(loc);

    } else {
      functionOp = dyn_cast<FuncOp>(existingOp);
    }

    rewriter.create<CallOp>(loc, functionOp, operands);

    rewriter.replaceOp(originalOp, {});
    return success();
  }
};

void QUIRToPulsePass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<PulseDialect>();
  registry.insert<complex::ComplexDialect>();
}

void QUIRToPulsePass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  setup_ = getCachedAnalysis<qu::LegacyInputConversion>();
  if (!setup_) {
    llvm::errs() << "Cannot initialize a conversion to pulse without a system "
                    "definition. This "
                    "error usually indicates that the system creation pass has "
                    "not been run prior to this point.";
  }

  auto symbolTable = SymbolTable(moduleOp);
  auto symbolTableRef = std::ref(symbolTable);

  QUIRTypeConverter typeConverter;
  ConversionTarget target(getContext());

  target.addLegalDialect<PulseDialect>();
  target.addLegalDialect<StandardOpsDialect>();
  target.addLegalDialect<BuiltinDialect>();
  target.addLegalDialect<complex::ComplexDialect>();
  target.addIllegalDialect<quir::QUIRDialect>();

  target.addLegalOp<quir::ConstantOp>();

  RewritePatternSet patterns(&getContext());
  populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
                                                           typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);

  patterns.add<DeclareQubitConversionPat>(&getContext(), typeConverter,
                                          setup_.getValue());
  patterns.add<DefCalConversionPat>(&getContext(), typeConverter,
                                    setup_.getValue(), symbolTableRef);

  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
    signalPassFailure();
}

llvm::StringRef QUIRToPulsePass::getArgument() const { return "quir-to-pulse"; }
llvm::StringRef QUIRToPulsePass::getDescription() const {
  return "Convert QUIR to Pulse.";
}
} // namespace mlir::pulse
