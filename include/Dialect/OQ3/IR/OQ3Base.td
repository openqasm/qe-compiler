//===- OQ3Dialect.td - MLIR OpenQASM 3 dialect -------------*- tablegen -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file defines the OpenQASM 3 MLIR dialect, its base operators, and
/// attributes.
///
//===----------------------------------------------------------------------===//

#ifndef OQ3_BASE_H
#define OQ3_BASE_H

//===----------------------------------------------------------------------===//
// OpenQASM 3 dialect definition
//===----------------------------------------------------------------------===//

def OQ3_Dialect : Dialect {
    let name = "oq3";
    let summary = "The OpenQASM 3 MLIR dialect.";
    let description = [{
        This is a high-level dialect that contains a mix of classical and
        quantum abstractions.
    }];
    let cppNamespace = "::mlir::oq3";
}

//===----------------------------------------------------------------------===//
// Base class for OpenQASM 3 dialect operations
//===----------------------------------------------------------------------===//

class OQ3_Op<string mnemonic, list<Trait> traits = []> :
        Op<OQ3_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Base class for OpenQASM 3 dialect attributes
//===----------------------------------------------------------------------===//

class OQ3_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<OQ3_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // OQ3_BASE_H
