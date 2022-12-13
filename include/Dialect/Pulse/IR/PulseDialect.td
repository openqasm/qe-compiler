//===- PulseDialect.td - Pulse dialect ---------------------*- tablegen -*-===//
//
// (C) Copyright IBM 2021.
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

include "PulseTypes.td"
include "PulseOps.td"

#endif // PULSE_DIALECT
