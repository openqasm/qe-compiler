//===- QCSAttributes.h - QCS dialect attributes -----------------*- C++ -*-===//
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
///  This file declares Quantum Control System dialect specific attributes.
///
//===----------------------------------------------------------------------===//

#ifndef DIALECT_QCS_QCSATTRIBUTES_H_
#define DIALECT_QCS_QCSATTRIBUTES_H_

#include "llvm/ADT/StringRef.h"

namespace mlir::qcs {
static inline llvm::StringRef getShotLoopAttrName() { return "qcs.shot_loop"; }
static inline llvm::StringRef getNumShotsAttrName() { return "qcs.num_shots"; }
} // namespace mlir::qcs

#endif // DIALECT_QCS_QCSATTRIBUTES_H_
