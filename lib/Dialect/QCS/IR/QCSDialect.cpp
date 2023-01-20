//===- QCSDialect.cpp - Quantum Control System dialect ----------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
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
#include "Dialect/QCS/IR/QCSAttributes.h"
#include "Dialect/QCS/IR/QCSOps.h"
#include "Dialect/QCS/IR/QCSTypes.h"

using namespace mlir;
using namespace mlir::qcs;

/// Tablegen Definitions
#include "Dialect/QCS/IR/QCSOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Quantum Control System dialect
//===----------------------------------------------------------------------===//

void QCSDialect::initialize() {

  addOperations<
#define GET_OP_LIST
#include "Dialect/QCS/IR/QCSOps.cpp.inc"
      >();
}
