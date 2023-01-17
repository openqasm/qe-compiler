//===- SystemOps.h - System dialect ops -------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#ifndef SYSTEM_SYSTEMOPS_H
#define SYSTEM_SYSTEMOPS_H

#include "Dialect/System/IR/SystemAttributes.h"
#include "Dialect/System/IR/SystemInterfaces.h"
#include "Dialect/System/IR/SystemTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

#include <set>
#include <vector>

#define GET_OP_CLASSES
#include "Dialect/System/IR/System.h.inc"

#endif // SYSTEM_SYSTEMOPS_H
