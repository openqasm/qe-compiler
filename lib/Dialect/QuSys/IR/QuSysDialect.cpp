//===- QuSysDialect.cpp - Quantum System dialect ----------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file defines the Quantum System dialect in MLIR.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QuSys/IR/QuSysDialect.h"
#include "Dialect/QuSys/IR/QuSysAttributes.h"
#include "Dialect/QuSys/IR/QuSysOps.h"
#include "Dialect/QuSys/IR/QuSysTypes.h"

using namespace mlir;
using namespace mlir::qusys;

/// Tablegen Definitions
#include "Dialect/QuSys/IR/QuSysOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Quantum System dialect.
//===----------------------------------------------------------------------===//

void QuSysDialect::initialize() {

  addOperations<
#define GET_OP_LIST
#include "Dialect/QuSys/IR/QuSysOps.cpp.inc"
      >();
}
