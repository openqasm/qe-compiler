//===- OQ3Dialect.h - OpenQASM 3 dialect ------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file declares the OpenQASM 3 dialect in MLIR.
///
//===----------------------------------------------------------------------===//

#ifndef DIALECT_OQ3_OQ3DIALECT_H_
#define DIALECT_OQ3_OQ3DIALECT_H_

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

#include "Dialect/OQ3/IR/OQ3OpsDialect.h.inc"

#endif // DIALECT_OQ3_OQ3DIALECT_H_
