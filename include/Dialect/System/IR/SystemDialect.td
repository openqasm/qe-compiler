//===- SystemDialect.td - System dialect -----------------------------*- tablegen -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===--------------------------------------------------------------------------------===//

#ifndef SYSTEM_DIALECT
#define SYSTEM_DIALECT

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/OpBase.td"

//===--------------------------------------------------------------------------------===//
// System dialect definition.
//===--------------------------------------------------------------------------------===//

def System_Dialect : Dialect {
    let name = "sys";
    let summary = "The System MLIR dialect.";
    let description = [{
        This is a low-level dialect that sits directly above target-dependent
        code generation. As such, the `System` dialect has no conception of, e.g.,
        a 'qubit' or 'angle'. The `System` dialect aims to provide fundamental types
        and operations that can be used at higher levels; e.g., OpenQASM 3, QUIR, and
        Pulse dilects.
    }];
    let cppNamespace = "::mlir::sys";
    let dependentDialects = [
        "mlir::arith::ArithmeticDialect",
        "mlir::scf::SCFDialect",
        "mlir::LLVM::LLVMDialect"
    ];
    // let hasConstantMaterializer = 1;
}

//===--------------------------------------------------------------------------------===//
// Base System operation definition.
//===--------------------------------------------------------------------------------===//

class SYS_Op<string mnemonic, list<Trait> traits = []> :
        Op<System_Dialect, mnemonic, traits>;

//===--------------------------------------------------------------------------------===//
// Base System attribute definition.
//===--------------------------------------------------------------------------------===//

class SYS_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<System_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

include "Dialect/System/IR/SystemTypes.td"
include "Dialect/System/IR/SystemTypeConstraints.td"
include "Dialect/System/IR/SystemAttributes.td"
include "Dialect/System/IR/SystemOps.td"

#endif // SYSTEM_DIALECT
