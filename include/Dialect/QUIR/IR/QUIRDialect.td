//===- QuirDialect.td - Quir dialect -----------------------*- tablegen -*-===//
//
// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
        "mlir::qcs::QCSDialect",
        "mlir::arith::ArithmeticDialect",
        "mlir::scf::SCFDialect",
        "mlir::LLVM::LLVMDialect"
    ];
    let hasConstantMaterializer = 1;
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
