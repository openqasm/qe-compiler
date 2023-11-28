//===- QUIRToLLVM.h - Convert QUIR to LLVM Dialect --------------*- C++ -*-===//
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
//  This file declares the pass for converting QUIR to LLVM dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CONVERSION_QUIRTOLLVM_QUIRTOLLVM_H_
#define CONVERSION_QUIRTOLLVM_QUIRTOLLVM_H_

#include "Conversion/QUIRToStandard/SwitchOpLowering.h"
#include "Dialect/QUIR/IR/QUIRDialect.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Error.h"

namespace mlir::quir {
static auto translateModuleToLLVMDialect(mlir::ModuleOp op,
                                         llvm::DataLayout &dataLayout)
    -> llvm::Error {

  // The MLIR documentation on lowering to LLVM IR recommends to use the
  // conversion patterns for all used dialects collected together, instead of
  // using the existing passes that treat only individual dialects.
  // https://mlir.llvm.org/docs/TargetLLVMIR/
  //
  // The Toy tutorial is the best documentation resource for the overall flow
  // https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/
  auto *context = op.getContext();
  assert(context);

  // Register LLVM dialect and all infrastructure required for translation to
  // LLVM IR
  mlir::registerBuiltinDialectTranslation(*context);
  mlir::registerLLVMDialectTranslation(*context);

  mlir::LLVMConversionTarget target(*context);
  target.addLegalDialect<mlir::LLVM::LLVMDialect>();
  target.addIllegalDialect<mlir::quir::QUIRDialect>();
  target.addLegalOp<mlir::ModuleOp>();

  mlir::LowerToLLVMOptions options(context);

  options.overrideIndexBitwidth(64);
  options.dataLayout = dataLayout;

  mlir::LLVMTypeConverter typeConverter(context, options);

  mlir::RewritePatternSet patterns(context);
  mlir::quir::populateSwitchOpLoweringPatterns(patterns);
  mlir::populateAffineToStdConversionPatterns(patterns);
  mlir::populateSCFToControlFlowConversionPatterns(patterns);
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                        patterns);
  mlir::populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  if (mlir::applyFullConversion(op, target, std::move(patterns)).failed())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to convert module to LLVM Dialect.");

  op->setAttr(
      mlir::LLVM::LLVMDialect::getDataLayoutAttrName(),
      mlir::StringAttr::get(context, dataLayout.getStringRepresentation()));

  return llvm::Error::success();
}
} // namespace mlir::quir

#endif // CONVERSION_QUIRTOLLVM_QUIRTOLLVM_H_
