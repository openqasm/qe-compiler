//===- OQ3Dialect.cpp - OpenQASM 3 dialect ----------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file defines the OpenQASM 3 dialect in MLIR.
///
//===----------------------------------------------------------------------===//

#include "Dialect/OQ3/IR/OQ3Dialect.h"
#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/OQ3/IR/OQ3Types.h"

using namespace mlir;
using namespace mlir::oq3;

/// Tablegen Definitions
#include "Dialect/OQ3/IR/OQ3OpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// OpenQASM 3 dialect
//===----------------------------------------------------------------------===//

void OQ3Dialect::initialize() {

  addOperations<
#define GET_OP_LIST
#include "Dialect/OQ3/IR/OQ3Ops.cpp.inc"
      >();
}
