//===- QCSDialect.td - MLIR Quantum Control System dialect -*- tablegen -*-===//
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
/// This file defines the Quantum Control System MLIR dialect, its base
/// operators, and attributes.
///
//===----------------------------------------------------------------------===//

#ifndef QCS_BASE_H
#define QCS_BASE_H

//===----------------------------------------------------------------------===//
// Quantum Control System dialect definition
//===----------------------------------------------------------------------===//

def QCS_Dialect : Dialect {
    let name = "qcs";
    let summary = "The Quantum Control System MLIR dialect.";
    let description = [{
        This is a low-level dialect that sits directly above target-dependent
        code generation. As such, the `QCS` dialect has no conception of,
        e.g., a 'qubit' or 'angle'. The `QCS` dialect aims to provide
        fundamental types and operations that can be used at higher levels;
        e.g., OpenQASM 3, QUIR, and Pulse dilects.
    }];
    let cppNamespace = "::mlir::qcs";
}

//===----------------------------------------------------------------------===//
// Base class for Quantum Control System dialect operations
//===----------------------------------------------------------------------===//

class QCS_Op<string mnemonic, list<Trait> traits = []> :
        Op<QCS_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Base class for Quantum Control System dialect attributes
//===----------------------------------------------------------------------===//

class QCS_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<QCS_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // QCS_BASE_H
