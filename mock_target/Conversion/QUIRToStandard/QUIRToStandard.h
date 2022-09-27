//===- QUIRToStd.h - Convert QUIR to Std Dialect ----------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  This file declares the pass for converting QUIR to std dialect
//
//===----------------------------------------------------------------------===//

#ifndef MOCK_CONVERSION_QUIRTOSTD_H
#define MOCK_CONVERSION_QUIRTOSTD_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace qssc::targets::mock::conversion {
struct MockQUIRToStdPass
    : public mlir::PassWrapper<MockQUIRToStdPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
};
} // namespace qssc::targets::mock::conversion

#endif // MOCK_CONVERSION_QUIRTOSTD_H
