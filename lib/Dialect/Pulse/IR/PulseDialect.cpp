//===- PulseDialect.cpp - Pulse dialect -------------------------*- C++ -*-===//
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
///  This file defines the Pulse dialect in MLIR.
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/IR/PulseDialect.h"



/// Tablegen Definitions

#define GET_TYPEDEF_CLASSES

#define GET_ATTRDEF_CLASSES

namespace mlir::pulse {

void pulse::PulseDialect::initialize() {

  addTypes<
#define GET_TYPEDEF_LIST
     >();

  addOperations<
#define GET_OP_LIST
     >();

  addAttributes<
#define GET_ATTRDEF_LIST
     >();
}

} // namespace mlir::pulse
