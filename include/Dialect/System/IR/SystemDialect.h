//===- SystemDialect.h - System dialect -------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#ifndef SYSTEM_SYSTEMDIALECT_H
#define SYSTEM_SYSTEMDIALECT_H

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"

#include <set>

#include "Dialect/System/IR/SystemDialect.h.inc"

#endif // SYSTEM_SYSTEMDIALECT_H
