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
//
//  This file declares the pass for converting QUIR to Pulse dialect
//
//===----------------------------------------------------------------------===//

#ifndef PULSE_CONVERSION_QUIRTOPULSE_H
#define PULSE_CONVERSION_QUIRTOPULSE_H

#include "Utils/LegacyInputConversion.h"
#include "Utils/SystemNodes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <functional>
#include <memory>

namespace mlir::pulse {

void processPlayOps(const std::shared_ptr<::qssc::utils::PlayOp> &play,
                    Location loc, MLIRContext *ctx, Value target,
                    OpBuilder builder);

class QUIRTypeConverter : public TypeConverter {
public:
  using TypeConverter::TypeConverter;
  QUIRTypeConverter();
};

class QUIRToPulsePass
    : public PassWrapper<QUIRToPulsePass, OperationPass<ModuleOp>> {
public:
  void getDependentDialects(DialectRegistry &registry) const override;
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;

protected:
  Optional<std::reference_wrapper<qssc::utils::LegacyInputConversion>> setup_;
};

} // namespace mlir::pulse

#endif // PULSE_CONVERSION_QUIRTOPULSE_H
