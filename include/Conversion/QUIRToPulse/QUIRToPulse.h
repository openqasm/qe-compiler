//===- QUIRToPulse.h - Convert QUIR to Pulse Dialect ------------*- C++ -*-===//
//
// (C) Copyright IBM 2021 - 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
