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


namespace {

} // anonymous namespace


namespace mlir::quir {
struct QUIRConvertDurationUnitsPass
    : public PassWrapper<QUIRConvertDurationUnitsPass, OperationPass<>> {

  Option<TimeUnits> units{*this, "units",
                        llvm::cl::desc("Target units to convert to"),
                        llvm::cl::values(
                          llvm::cl::clEnumValue(TimeUnits::dt, "dt", "Scheduling sample rate"),
                          llvm::cl::clEnumValue(TimeUnits::s, "s", "seconds"),
                          llvm::cl::clEnumValue(TimeUnits::ms, "ms", "milliseconds"),
                          llvm::cl::clEnumValue(TimeUnits::us, "us", "microseconds"),
                          llvm::cl::clEnumValue(TimeUnits::ns, "ns", "nanoseconds"),
                          llvm::cl::clEnumValue(TimeUnits::ps, "ps", "picoseconds"),
                          llvm::cl::clEnumValue(TimeUnits::fs, "fs", "femtoseconds")),
                        llvm::cl::value_desc("enum"), llvm::cl::init(TimeUnits::dt)};

  Option<double> dtDuration{*this, "dt-duration",
                        llvm::cl::desc("Duration of dt (scheduling cycle) in seconds."),
                        llvm::cl::value_desc("num"), llvm::cl::init(-1)};


  void runOnOperation() override;

  TimeUnits getTargetConvertUnits() const;
  double getDTDuration();
  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;

}; // struct QUIRConvertDurationUnitsPass

} // end namespace mlir::quir

#endif // QUIR_CONVERT_DURATION_UNITS_H
