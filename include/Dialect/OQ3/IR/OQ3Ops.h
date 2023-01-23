//===- OQ3Ops.h - OpenQASM 3 dialect ops ------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file declares the operations in the OpenQASM 3 dialect.
///
//===----------------------------------------------------------------------===//

#ifndef DIALECT_OQ3_OQ3OPS_H_
#define DIALECT_OQ3_OQ3OPS_H_

#include "Dialect/OQ3/IR/OQ3Types.h"

#include "mlir/IR/BuiltinOps.h"

#include <set>
#include <vector>

#define GET_OP_CLASSES
#include "Dialect/OQ3/IR/OQ3Ops.h.inc"

#endif // DIALECT_OQ3_OQ3OPS_H_
