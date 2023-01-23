//===- QuirDialect.h - Quir dialect -----------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file declares the QUIR dialect in MLIR.
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_QUIRDIALECT_H
#define QUIR_QUIRDIALECT_H

#include "Dialect/QCS/IR/QCSDialect.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

#include <set>

#include "Dialect/QUIR/IR/QUIRDialect.h.inc"

#endif // QUIR_QUIRDIALECT_H
