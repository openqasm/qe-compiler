//===- OutputClassicalRegisters.cpp ----------------------------*- C++ -*-===//
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
//  This file implements passes to output classical register values
//
//===----------------------------------------------------------------------===//
#include "Conversion/OutputClassicalRegisters.h"
#include "Conversion/QUIRToStandard/VariablesToGlobalMemRefConversion.h"

#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/QCS/IR/QCSOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace qssc::targets::simulator::conversion {

class OutputCRegsPassImpl {
public:
  OutputCRegsPassImpl() = default;

  void runOnOperation(MLIRContext *context, ModuleOp moduleOp) {
    mlir::TypeConverter typeConverter;
    ConversionTarget target(*context);

    RewritePatternSet patterns(context);

    prepareConversion(moduleOp);
    insertOutputCRegs(moduleOp);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns))))
      llvm::outs() << "[OutputCRegsPass] Failed applyPartialConversion\n";
  }

private:
  void prepareConversion(ModuleOp moduleOp) {
    // Create definition table that maps classical register names ->
    // defining operations.
    cbitDecls.clear();
    moduleOp->walk([&](oq3::DeclareVariableOp op) {
      if (auto ty = op.type().dyn_cast<quir::CBitType>())
        cbitDecls.emplace_back(op.getName().str(), op);
    });

    // Declare the printf function.
    // TODO: In future, the result values should be printed with another way.
    OpBuilder builder(moduleOp);
    builder.setInsertionPointToStart(moduleOp.getBody());
    const auto i32Ty = builder.getI32Type();
    const auto i8PtrTy = LLVM::LLVMPointerType::get(builder.getI8Type());
    const auto printfTy =
        LLVM::LLVMFunctionType::get(i32Ty, i8PtrTy, /*isVarArgs=*/true);
    printfFuncOp = builder.create<LLVM::LLVMFuncOp>(moduleOp->getLoc(),
                                                    "printf", printfTy);
  }

  // Insert output ops where `qcf.finalize` is called.
  // We do not erase `qcf.finalize` because a subsequent pass may use it
  // for the translation.
  void insertOutputCRegs(ModuleOp moduleOp) {
    // Assume that `qcf.finalize` is called only once.
    moduleOp->walk([&](qcs::SystemFinalizeOp op) {
      OpBuilder builder(op);

      // Define constant strings for printing globally.
      if (globalStrs.find("\n") == globalStrs.end()) {
        const auto varName = std::string{"str_endline"};
        const auto value = std::string{"\n\0", 2};
        globalStrs["\n"] = LLVM::createGlobalString(
            op->getLoc(), builder, varName, value, LLVM::Linkage::Private);
      }
      if (globalStrs.find("%d") == globalStrs.end()) {
        const auto varName = std::string{"str_digit"};
        const auto value = std::string{"%d\0", 3};
        globalStrs["%d"] = LLVM::createGlobalString(
            op->getLoc(), builder, varName, value, LLVM::Linkage::Private);
      }

      // Print the values of classical registers in the declared order.
      for (auto &[name, declOp] : cbitDecls) {
        if (globalStrs.find(name) == globalStrs.end()) {
          const auto varName = std::string{"str_creg_"} + name;
          const auto value = std::string{"  "} + name + std::string{" : \0", 4};
          globalStrs[name] = LLVM::createGlobalString(
              op->getLoc(), builder, varName, value, LLVM::Linkage::Private);
        }

        builder.create<LLVM::CallOp>(op->getLoc(), printfFuncOp,
                                     globalStrs[name]);
        auto cBitTy = declOp.type().dyn_cast<quir::CBitType>();
        const int width = cBitTy.getWidth();
        const auto boolTy = builder.getI1Type();
        auto loaded =
            builder.create<oq3::VariableLoadOp>(op->getLoc(), cBitTy, name);
        for (int i = width - 1; i >= 0; --i) {
          auto indexAttr = builder.getIndexAttr(i);
          auto bit = builder.create<oq3::CBitExtractBitOp>(op->getLoc(), boolTy,
                                                           loaded, indexAttr);
          builder.create<LLVM::CallOp>(op->getLoc(), printfFuncOp,
                                       ValueRange{globalStrs["%d"], bit});
        }
        builder.create<LLVM::CallOp>(op->getLoc(), printfFuncOp,
                                     globalStrs["\n"]);
      }
    });
  }

private:
  std::vector<std::pair<std::string, oq3::DeclareVariableOp>> cbitDecls;
  std::map<std::string, mlir::Value> globalStrs;
  LLVM::LLVMFuncOp printfFuncOp;
};

OutputCRegsPass::OutputCRegsPass() : PassWrapper() {
  impl = std::make_shared<OutputCRegsPassImpl>();
}

void OutputCRegsPass::runOnOperation(SimulatorSystem &system) {
  auto *context = &getContext();
  auto moduleOp = getOperation();
  impl->runOnOperation(context, moduleOp);
}

void OutputCRegsPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<LLVM::LLVMDialect, mlir::memref::MemRefDialect,
                  mlir::AffineDialect, arith::ArithmeticDialect>();
}

llvm::StringRef OutputCRegsPass::getArgument() const {
  return "simulator-output-cregs";
}

llvm::StringRef OutputCRegsPass::getDescription() const {
  return "Insert output ops for classical registers";
}

} // namespace qssc::targets::simulator::conversion
