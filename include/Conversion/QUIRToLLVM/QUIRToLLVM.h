//===- QUIRToLLVM.h - Convert QUIR to LLVM Dialect --------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
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
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/IR/BuiltinOps.h"
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
  mlir::populateLoopToStdConversionPatterns(patterns);
  mlir::quir::populateSwitchOpLoweringPatterns(patterns);
  mlir::arith::populateArithmeticToLLVMConversionPatterns(typeConverter,
                                                          patterns);
  mlir::populateAffineToStdConversionPatterns(patterns);
  mlir::populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  mlir::populateStdToLLVMFuncOpConversionPattern(typeConverter, patterns);
  mlir::populateStdToLLVMConversionPatterns(typeConverter, patterns);

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
