//===- SystemOps.cpp - System dialect ops -----------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file defines the operations in the Quantum System dialect.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QuSys/IR/QuSysOps.h"
#include "Dialect/QuSys/IR/QuSysDialect.h"
#include "Dialect/QuSys/IR/QuSysTypes.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace mlir::qusys;

#define GET_OP_CLASSES
#include "Dialect/QuSys/IR/QuSysOps.cpp.inc"
