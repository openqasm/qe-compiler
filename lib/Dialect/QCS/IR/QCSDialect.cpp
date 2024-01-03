//===- QCSDialect.cpp - Quantum Control System dialect ----------*- C++ -*-===//
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
///  This file defines the Quantum Control System dialect in MLIR.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QCS/IR/QCSDialect.h"

// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/QCS/IR/QCSAttributes.h"
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/QCS/IR/QCSOps.h"
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/QCS/IR/QCSTypes.h"

using namespace mlir;
using namespace mlir::qcs;

/// Tablegen Definitions
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/QCS/IR/QCSOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Quantum Control System dialect
//===----------------------------------------------------------------------===//

void QCSDialect::initialize() {

  addOperations<
#define GET_OP_LIST
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/QCS/IR/QCSOps.cpp.inc"
      >();
}
