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

namespace {
    
// Maintain classical register declarations in the declared order.
std::vector<std::pair<std::string, oq3::DeclareVariableOp>> cbitDecls;
std::map<std::string, mlir::Value> globalStrs;
LLVM::LLVMFuncOp printfFuncOp;

void collectCBitDecls(ModuleOp moduleOp) {
  cbitDecls.clear();
  
  moduleOp->walk([&](oq3::DeclareVariableOp op) {
    if(auto ty = op.type().dyn_cast<quir::CBitType>()) {
      cbitDecls.emplace_back(op.getName().str(), op);
    }
  });
}

void prepareConversion(ModuleOp moduleOp) {
  collectCBitDecls(moduleOp);
  
  OpBuilder builder(moduleOp);
  builder.setInsertionPointToStart(moduleOp.getBody());
  
  // printf
  const auto i32Ty = builder.getIntegerType(32);
  const auto i8PtrTy = LLVM::LLVMPointerType::get(builder.getIntegerType(8));
  const auto printfTy = LLVM::LLVMFunctionType::get(i32Ty, i8PtrTy, /*isVarArgs=*/true);
  printfFuncOp = builder.create<LLVM::LLVMFuncOp>(moduleOp->getLoc(), "printf", printfTy);
}
    
}

namespace qssc::targets::simulator::conversion {

// Insert output ops where `qcf.finalize` is called.
// The reason why `qcf.finalize` is left (that is don't use a conversion pattern)
// is that a subsequent pass can translate it properly.
void insertOutputCRegs(ModuleOp moduleOp) {
  moduleOp->walk([&] (qcs::SystemFinalizeOp op) {
    OpBuilder builder(op);

    if(globalStrs.find("\n") == globalStrs.end()) {
      const auto varName = std::string{"str_endline"};
      const auto value = std::string{"\n\0", 2};
      globalStrs["\n"] = LLVM::createGlobalString(
        op->getLoc(), builder, varName, value, LLVM::Linkage::Private);
    }
    if(globalStrs.find("%d") == globalStrs.end()) {
      const auto varName = std::string{"str_digit"};
      const auto value = std::string{"%d\0", 3};
      globalStrs["%d"] = LLVM::createGlobalString(
        op->getLoc(), builder, varName, value, LLVM::Linkage::Private);
    }
    for(auto& [name, declOp] : cbitDecls) {
      if(globalStrs.find(name) == globalStrs.end()) {
        const auto varName = std::string{"str_creg_"} + name;
        const auto value = std::string{"  "} + name + std::string{" : \0", 4};
        globalStrs[name] = LLVM::createGlobalString(
            op->getLoc(), builder, varName, value, LLVM::Linkage::Private);
      }

      builder.create<LLVM::CallOp>(op->getLoc(), printfFuncOp, globalStrs[name]);
      auto cBitTy = declOp.type().dyn_cast<quir::CBitType>();
      const int width = cBitTy.getWidth();
      const auto boolTy = builder.getIntegerType(1);
      auto loaded = builder.create<oq3::VariableLoadOp>(op->getLoc(), cBitTy, name);
      for(int i = width - 1; i >= 0; --i) {
        auto indexAttr = builder.getIndexAttr(i);
        auto bit = builder.create<oq3::CBitExtractBitOp>(op->getLoc(), boolTy, loaded, indexAttr);
        builder.create<LLVM::CallOp>(
            op->getLoc(), printfFuncOp, ValueRange{globalStrs["%d"], bit});
      }
      builder.create<LLVM::CallOp>(
          op->getLoc(), printfFuncOp, globalStrs["\n"]);
    }
  });
}


void OutputCRegsPass::runOnOperation(SimulatorSystem &system) {
  ModuleOp moduleOp = getOperation();

  mlir::TypeConverter typeConverter; // TODO
  auto *context = &getContext();
  ConversionTarget target(*context);
  
  RewritePatternSet patterns(context);

  prepareConversion(moduleOp);
  insertOutputCRegs(moduleOp);

  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    llvm::outs() << "[OutputCRegsPass] Failed applyPartialConversion\n";
  }
}

void OutputCRegsPass::getDependentDialects(
    DialectRegistry &registry) const {
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
