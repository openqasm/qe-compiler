//===- QUIRTraits.td - QUIR dialect traits -*- C++ -*-========================//
//
// (C) Copyright IBM 2022, 2023.
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
///  Traits for the QUIR dialect
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_TRAITS
#define QUIR_TRAITS

include "mlir/IR/OpBase.td"

// QUIR Traits

// Operation applies a quantum channel to the quantum system.
// Ie., it is a completely positive trace perserving (CPTP) map
def CPTPOp : NativeOpTrait<"CPTPOp">{
    let cppNamespace= "::mlir::quir";
}

// Operation applies a unitary transformation to the quantum system.
// Is a subclass of CPTPOp.
def UnitaryOp : NativeOpTrait<"UnitaryOp"> {
   let cppNamespace= "::mlir::quir";
}

#endif // QUIR_TRAITS
