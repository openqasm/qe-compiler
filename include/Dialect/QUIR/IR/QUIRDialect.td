//===- QuirDialect.td - Quir dialect -----------------------*- tablegen -*-===//
//
// (C) Copyright IBM 2021.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#ifndef QUIR_DIALECT
#define QUIR_DIALECT

include "mlir/Interfaces/CallInterfaces.td"
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
        "mlir::arith::ArithmeticDialect",
        "mlir::scf::SCFDialect",
        "mlir::LLVM::LLVMDialect"
    ];
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

include "Dialect/QUIR/IR/QUIRTypes.td"
include "Dialect/QUIR/IR/QUIRTypeConstraints.td"
include "Dialect/QUIR/IR/QUIRAttributes.td"
include "Dialect/QUIR/IR/QUIRTraits.td"
include "Dialect/QUIR/IR/QUIROps.td"
include "Dialect/QUIR/IR/QUIRMathOps.td"

#endif // QUIR_DIALECT
