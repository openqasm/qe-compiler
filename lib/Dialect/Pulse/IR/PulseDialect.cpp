//===- PulseDialect.cpp - Pulse dialect -------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
///
///  This file defines the Pulse dialect in MLIR.
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/Pulse/IR/PulseOps.h"
#include "Dialect/Pulse/IR/PulseTypes.h"

#include "mlir/IR/Dialect.h"

#include "llvm/ADT/TypeSwitch.h"

/// Tablegen Definitions
#include "Dialect/Pulse/IR/PulseDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/Pulse/IR/PulseTypes.cpp.inc"

namespace mlir::pulse {

void pulse::PulseDialect::initialize() {

  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/Pulse/IR/PulseTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "Dialect/Pulse/IR/Pulse.cpp.inc"
      >();
}

} // namespace mlir::pulse
