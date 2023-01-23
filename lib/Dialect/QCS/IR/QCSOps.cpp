//===- QCSOps.cpp - Quantum Control System dialect ops ----------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file defines the operations in the Quantum Control System dialect.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QCS/IR/QCSOps.h"
#include "Dialect/QCS/IR/QCSDialect.h"
#include "Dialect/QCS/IR/QCSTypes.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace mlir::qcs;

#define GET_OP_CLASSES
#include "Dialect/QCS/IR/QCSOps.cpp.inc"
