//===- QCSOps.cpp - Quantum Control System dialect ops ----------*- C++ -*-===//
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
/// This file defines the operations in the Quantum Control System dialect.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QCS/IR/QCSOps.h"

#include "Dialect/QCS/IR/QCSTypes.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include <mlir/IR/OpImplementation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Support/LogicalResult.h>

#include <cassert>

using namespace mlir;
using namespace mlir::qcs;

#define GET_OP_CLASSES
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/QCS/IR/QCSOps.cpp.inc"

//===----------------------------------------------------------------------===//
// ParameterLoadOp
//===----------------------------------------------------------------------===//

// Returns the float value from the initial value of this parameter
ParameterType ParameterLoadOp::getInitialValue() {
  auto *op = getOperation();
  double retVal = 0.0;
  if (op->hasAttr("initialValue")) {
    auto initAttr = op->getAttr("initialValue").dyn_cast<FloatAttr>();
    if (initAttr)
      retVal = initAttr.getValue().convertToDouble();
  }
  return retVal;
}

//===----------------------------------------------------------------------===//
// End ParameterLoadOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ParallelEndOp
//===----------------------------------------------------------------------===//

mlir::ParseResult ParallelEndOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  return mlir::success();
}

void ParallelEndOp::print(mlir::OpAsmPrinter &printer) {
  printer << getOperationName();
}
