//===- QUIRTypes.td - QUIR dialect types -------------------*- tablegen -*-===//
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

#ifndef QUIR_TYPES
#define QUIR_TYPES

include "Dialect/QUIR/IR/QUIREnums.td"

include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/IR/OpBase.td"

class QUIR_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<QUIRDialect, name, traits> {
        let mnemonic = typeMnemonic;
}

def QUIR_QubitType : QUIR_Type<"Qubit", "qubit"> {
    let summary = "Qubit register with arbitrary width.";
    let description = [{
        Qubit types define a single qubit or a qubit register with a designated
        width.
    }];

    let parameters = (ins "int":$width);
    let assemblyFormat = "`<` $width `>`";
    let genVerifyDecl = 1;
}

def QUIR_CBitType : QUIR_Type<"CBit", "cbit", [MemRefElementTypeInterface]> {
    let summary = "Classical bit register with arbitrary width.";
    let description = [{
        CBit types define a single classical bit or a classical bit register
        with a designated width.
    }];

    let parameters = (ins "unsigned":$width);
    let assemblyFormat = "`<` $width `>`";
}

def QUIR_AngleType : QUIR_Type<"Angle", "angle"> {
    let summary = "A classical angle.";
    let description = [{
        A QUIR Angle represents a classical angle as a fixed-point number.
    }];

    let parameters = (ins "Optional<int>":$width);
    // TODO in LLVM 15+, the optional width could be handled with assembly declarative syntax.
    // let assemblyFormat = "(`<` $width^ `>`)?";
    let genVerifyDecl = 1;
}



def QUIR_DurationType : QUIR_Type<"Duration", "duration"> {
    let summary = "represents a duration of time.";
    let description = [{
        A QUIR duration type.

        Example:
            ```mlir
                // Defaults to units at the scheduling rate (dt)
                %duration_dt = quir.constant #quir.duration<10.0 : !quir.duration>
                %duration_s = quir.constant #quir.duration<3.14 : !quir.duration<s>>
            ```

     }];

    // As best I an tell this is undocumented in LLVM 14.
    // Based on studying the implementation of EnumAttr
    // https://github.com/llvm/llvm-project/blob/f28c006a5895fc0e329fe15fead81e37457cb1d1/mlir/include/mlir/IR/EnumAttr.td#L86
    let parameters = (ins EnumParameter<TimeUnits>:$units);

    let assemblyFormat = "`<` $units `>`";
}

def QUIR_StretchType : QUIR_Type<"Stretch", "stretch"> {
    let summary = "A variable non-negative duration.";
    let description = [{
        Stretch variables can grow as necessary to satisfy constraints. They
        are resolved at compile time.
    }];
}

#endif // QUIR_TYPES
