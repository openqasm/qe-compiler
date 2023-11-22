//===- QuirDialect.td - Quir dialect -----------------------*- tablegen -*-===//
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

#ifndef QUIR_DIALECT
#define QUIR_DIALECT

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// QUIR dialect definition.
//===----------------------------------------------------------------------===//

def QUIRDialect : Dialect {
    let name = "quir";
    let summary = "A quantum circuit Intermediate Representation MLIR dialect.";
    let description = [{
        This dialect is targeted at modeling quantum circuits written in OpenQASM 3.0
    }];
    let cppNamespace = "::mlir::quir";
    let dependentDialects = [
        "mlir::oq3::OQ3Dialect",
        "mlir::qcs::QCSDialect",
        "mlir::arith::ArithDialect",
        "mlir::scf::SCFDialect",
        "mlir::LLVM::LLVMDialect"
    ];
    let hasConstantMaterializer = 1;
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Base quir operation definition.
//===----------------------------------------------------------------------===//

class QUIR_Op<string mnemonic, list<Trait> traits = []> :
        Op<QUIRDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Base quir attribute definition.
//===----------------------------------------------------------------------===//

class QUIR_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<QUIRDialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // QUIR_DIALECT
