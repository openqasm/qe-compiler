//===- QCSDialect.cpp - Quantum Control System dialect ----------*- C++ -*-===//
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
///  This file defines the Quantum Control System dialect in MLIR.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QCS/IR/QCSDialect.h"
#include "Dialect/QCS/IR/QCSAttributes.h"
#include "Dialect/QCS/IR/QCSOps.h"
#include "Dialect/QCS/IR/QCSTypes.h"

using namespace mlir;
using namespace mlir::qcs;

/// Tablegen Definitions
#include "Dialect/QCS/IR/QCSOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Quantum Control System dialect
//===----------------------------------------------------------------------===//

void QCSDialect::initialize() {

  addOperations<
#define GET_OP_LIST
#include "Dialect/QCS/IR/QCSOps.cpp.inc"
      >();
}
