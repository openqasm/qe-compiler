//===- SystemTypes.td - System dialect types ---------------*- tablegen -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#ifndef SYSTEM_TYPES
#define SYSTEM_TYPES

include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpBase.td"

class SYS_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<System_Dialect, name, traits> {
        let mnemonic = typeMnemonic;
}

#endif // SYSTEM_TYPES
