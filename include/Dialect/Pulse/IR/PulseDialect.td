//===- PulseDialect.td - Pulse dialect ---------------------*- tablegen -*-===//
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

include "Dialect/Pulse/IR/PulseTypes.td"
include "Dialect/Pulse/IR/PulseOps.td"

#endif // PULSE_DIALECT
