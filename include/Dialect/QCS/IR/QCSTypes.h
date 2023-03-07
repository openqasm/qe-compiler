//===- QCSTypes.h - Quantum Control System dialect types --*- C++ -*-=========//
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
/// This file declares the types in the Quantum Control System dialect.
///
//===----------------------------------------------------------------------===//

#ifndef DIALECT_QCS_QCSTYPES_H_
#define DIALECT_QCS_QCSTYPES_H_

// TODO: Temporary, until constraints between `OpenQASM3`, `QUIR`, `Pulse`, and
// `System` dialects are ironed out.
#include "Dialect/QUIR/IR/QUIRTypes.h"

namespace mlir::qcs {} // namespace mlir::qcs

#endif // DIALECT_QCS_QCSTYPES_H_
