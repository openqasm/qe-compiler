//===- QUIRTypes.td - QUIR dialect types -------------------*- tablegen -*-===//
//
// (C) Copyright IBM 2020, 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#ifndef QUIR_TYPES
#define QUIR_TYPES

include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpBase.td"

class OQ3_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<OQ3_Dialect, name, traits> {
        let mnemonic = typeMnemonic;
}

def OQ3_Float : OQ3_Type<"Float", "float">;

#endif // QUIR_TYPES
