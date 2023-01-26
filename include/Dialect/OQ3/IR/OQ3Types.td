//===- OQ3Types.td - OpenQASM 3 dialect types --------------*- tablegen -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file declares the OpenQASM 3 types.
///
//===----------------------------------------------------------------------===//

#ifndef OQ3_TYPES
#define OQ3_TYPES

// include "mlir/IR/BuiltinTypeInterfaces.td"
// include "mlir/IR/OpBase.td"

class OQ3_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<OQ3_Dialect, name, traits> {
        let mnemonic = typeMnemonic;
}

#endif // OQ3_TYPES
