//===- SystemOps.cpp - System dialect ops -----------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#include "Dialect/System/IR/SystemOps.h"
#include "Dialect/System/IR/SystemAttributes.h"
#include "Dialect/System/IR/SystemDialect.h"
#include "Dialect/System/IR/SystemTypes.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace mlir::sys;

#define GET_OP_CLASSES
#include "Dialect/System/IR/System.cpp.inc"
