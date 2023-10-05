//===- ConvertDurationUnits.h - Convert Duration Unis  ----------*- C++ -*-===//
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
///
///  This file declares the pass for converting the units of Durations
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_CONVERT_DURATION_UNITS_H
#define QUIR_CONVERT_DURATION_UNITS_H

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringRef.h"


namespace mlir::quir {
struct QUIRConvertDurationUnitsPass
    : public PassWrapper<QUIRConvertDurationUnitsPass, OperationPass<>> {
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;

}; // struct QUIRConvertDurationUnitsPass

} // end namespace mlir::quir

#endif // QUIR_CONVERT_DURATION_UNITS_H
