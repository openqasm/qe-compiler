//===- QuSysDialect.td - MLIR Quantum System dialect -------*- tablegen -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file defines the Quantum System MLIR dialect, its base operators, and
/// attributes.
///
//===----------------------------------------------------------------------===//

#ifndef DIALECTS_QUSYS_IR_BASE
#define DIALECTS_QUSYS_IR_BASE

//===----------------------------------------------------------------------===//
// Quantum System dialect definition
//===----------------------------------------------------------------------===//

def QuSys_Dialect : Dialect {
    let name = "qusys";
    let summary = "The Quantum System MLIR dialect.";
    let description = [{
        This is a low-level dialect that sits directly above target-dependent
        code generation. As such, the `QuSys` dialect has no conception of,
        e.g., a 'qubit' or 'angle'. The `QuSys` dialect aims to provide
        fundamental types and operations that can be used at higher levels;
        e.g., OpenQASM 3, QUIR, and Pulse dilects.
    }];
    let cppNamespace = "::mlir::qusys";
}

//===----------------------------------------------------------------------===//
// Base class for Quantum System dialect operations
//===----------------------------------------------------------------------===//

class QuSys_Op<string mnemonic, list<Trait> traits = []> :
        Op<QuSys_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Base class for Quantum System dialect attributes
//===----------------------------------------------------------------------===//

class QuSys_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<QuSys_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // DIALECTS_QUSYS_IR_BASE
