//===- SystemInterfaces.h - System dialect Interfaces -*- C++ -*-=============//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  Interfaces for the System dialect
///
//===----------------------------------------------------------------------===//

#ifndef SYSTEM_SYSTEMINTERFACES_H
#define SYSTEM_SYSTEMINTERFACES_H

#include "mlir/IR/OpDefinition.h"

#include <set>

//===----------------------------------------------------------------------===//
// Operation Interface Types
//===----------------------------------------------------------------------===//

namespace mlir::sys::interfaces_impl {} // namespace mlir::sys::interfaces_impl

//===----------------------------------------------------------------------===//
// Tablegen Interface Definitions
//===----------------------------------------------------------------------===//

#include "Dialect/System/IR/SystemInterfaces.h.inc"

#endif // SYSTEM_SYSTEMINTERFACES_H
