//===- QCSOps.h - Quantum Control System dialect ops ------------*- C++ -*-===//
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
/// This file declares the operations in the Quantum Control System dialect.
///
//===----------------------------------------------------------------------===//

#ifndef DIALECT_QCS_QCSOPS_H_
#define DIALECT_QCS_QCSOPS_H_

// TODO: move necessary components to `QCS`
#include "Dialect/QUIR/IR/QUIRInterfaces.h"
#include "Dialect/QUIR/IR/QUIRTraits.h"

#include "Dialect/QCS/IR/QCSTypes.h"

#define GET_OP_CLASSES
#include "Dialect/QCS/IR/QCSOps.h.inc"

#endif // DIALECT_QCS_QCSOPS_H_
