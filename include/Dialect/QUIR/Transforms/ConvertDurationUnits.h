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

#include "Dialect/QUIR/IR/QUIREnums.h"

#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"

namespace mlir::quir {

/// @brief This pass will convert all standard usage of durations
/// within QUIR to the input target units. It is useful for canonicalizing
/// durations within a program to uniform base unit such as the target
/// timestep "dt".
struct ConvertDurationUnitsPass
    : public PassWrapper<ConvertDurationUnitsPass, OperationPass<>> {

  ConvertDurationUnitsPass() = default;
  ConvertDurationUnitsPass(const ConvertDurationUnitsPass &pass)
      : PassWrapper(pass) {}
  ConvertDurationUnitsPass(const TimeUnits inUnits,
                           const double inDtTimestep = 1.) {
    units = inUnits;
    dtTimestep = inDtTimestep;
  }

  Option<TimeUnits> units{
      *this,
      "units",
      llvm::cl::desc("Target units to convert to"),
      llvm::cl::values(
          clEnumValN(TimeUnits::dt, "dt", "Scheduling sample rate"),
          clEnumValN(TimeUnits::s, "s", "seconds"),
          clEnumValN(TimeUnits::ms, "ms", "milliseconds"),
          clEnumValN(TimeUnits::us, "us", "microseconds"),
          clEnumValN(TimeUnits::ns, "ns", "nanoseconds"),
          clEnumValN(TimeUnits::ps, "ps", "picoseconds"),
          clEnumValN(TimeUnits::fs, "fs", "femtoseconds")),
      llvm::cl::value_desc("enum"),
      llvm::cl::init(TimeUnits::dt)};

  Option<double> dtTimestep{
      *this, "dt-timestep",
      llvm::cl::desc("Duration of dt (the scheduling timestep) in seconds. "
                     "Defaults to 1s."),
      llvm::cl::value_desc("num"), llvm::cl::init(1.)};

  void runOnOperation() override;

  virtual TimeUnits getTargetConvertUnits() const;
  virtual double getDtTimestep();
  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  llvm::StringRef getName() const override;
  std::string passName =
      "Convert Duration Units Pass (" + getArgument().str() + ")";

}; // struct ConvertDurationUnitsPass

} // end namespace mlir::quir

#endif // QUIR_CONVERT_DURATION_UNITS_H
