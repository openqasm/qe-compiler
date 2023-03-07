//===- PulseDialect.td - Pulse dialect ---------------------*- tablegen -*-===//
//
// (C) Copyright IBM 2021, 2023.
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

#ifndef PULSE_DIALECT
#define PULSE_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Pulse dialect definition.
//===----------------------------------------------------------------------===//

def PulseDialect : Dialect {
    let name = "pulse";
    let summary = "A pulse programming Intermediate Representation MLIR dialect.";
    let description = [{
        This dialect is targeted at modeling the implementation of microcoded circuit
        instructions written as microwave pulses. It is designed to be the underlying
        representation of QUIR gates.
    }];
    let cppNamespace = "::mlir::pulse";
}

//===----------------------------------------------------------------------===//
// Base Pulse operation definition.
//===----------------------------------------------------------------------===//

class Pulse_Op<string mnemonic, list<Trait> traits = []> :
        Op<PulseDialect, mnemonic, traits>;

include "PulseTypes.td"
include "PulseOps.td"

#endif // PULSE_DIALECT
