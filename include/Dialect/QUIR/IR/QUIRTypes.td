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

def QUIR_CBitType : QUIR_Type<"CBit", "cbit"> {
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
    let summary = "represents a duration of time as a string w/ specified units.";
    let description = [{
        A QUIR duration expresses timing.
     }];
}

def QUIR_StretchType : QUIR_Type<"Stretch", "stretch"> {
    let summary = "A variable non-negative duration.";
    let description = [{
        Stretch variables can grow as necessary to satisfy constraints. They
        are resolved at compile time.
    }];
}

#endif // QUIR_TYPES
