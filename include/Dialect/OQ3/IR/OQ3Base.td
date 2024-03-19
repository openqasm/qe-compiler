//===- OQ3Base.td - MLIR OpenQASM 3 dialect base -----------*- tablegen -*-===//
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

include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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
    let dependentDialects = [
        "mlir::arith::ArithDialect",
        "mlir::math::MathDialect",
        "mlir::LLVM::LLVMDialect"
    ];
}

//===----------------------------------------------------------------------===//
// OpenQASM 3 dialect base operations
//===----------------------------------------------------------------------===//

class OQ3_Op<string mnemonic, list<Trait> traits = []> :
    Op<OQ3_Dialect, mnemonic, traits>;

class OQ3_UnaryOp<string mnemonic, list<Trait> traits = []> :
      OQ3_Op<mnemonic, !listconcat(traits, [Pure])> {
  let arguments = (ins AnyClassical:$operand);
  let results = (outs AnyClassical:$result);
}

class OQ3_BinaryOp<string mnemonic, list<Trait> traits = []> :
    OQ3_Op<mnemonic, !listconcat(traits, [Pure])> {
  let arguments = (ins AnyClassical:$operand1, AnyClassical:$operand2);
  let results = (outs AnyClassical:$result);
  let assemblyFormat = [{
        attr-dict $lhs `,` $rhs `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Base class for OpenQASM 3 dialect attributes
//===----------------------------------------------------------------------===//

class OQ3_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<OQ3_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif // OQ3_BASE_H
