//===- QUIRPatterns.td - QUIR dialect declarative rewrite rules (patterns) --*--
// tablegen -*-===//
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

#ifndef QUIR_PATTERNS
#define QUIR_PATTERNS

include "mlir/Dialect/StandardOps/IR/Ops.td"
include "Dialect/QUIR/IR/QUIR.td"

def CbitNotNotPat : Pat<
    (QUIR_Cbit_NotOp (QUIR_Cbit_NotOp $in)), (replaceWithValue $in)>;

#endif // QUIR_PATTERNS
