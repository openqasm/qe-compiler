// RUN: qss-compiler -X=mlir --remove-unused-variables %s | FileCheck %s --check-prefix=UNUSED

//
// This code is part of Qiskit.
//
// (C) Copyright IBM 2023.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.


// UNUSED: quir.declare_variable @isUsed : !quir.cbit<1>
// UNUSED: quir.declare_variable {output} @isOutput : !quir.cbit<1>
// UNUSED-NOT: quir.declare_variable @storeOnly : !quir.cbit<1>
// UNUSED-NOT: quir.declare_variable @notUsed : !quir.cbit<1>
quir.declare_variable @isUsed : !quir.cbit<1>
quir.declare_variable {output} @isOutput : !quir.cbit<1>
quir.declare_variable @storeOnly : !quir.cbit<1>
quir.declare_variable @notUsed : !quir.cbit<1>
// UNUSED: func @variableTests
func @variableTests(%ref : memref<1xi1>, %ind : index) {
    %false = arith.constant false
    %false_cbit = "quir.cast"(%false) : (i1) -> !quir.cbit<1>

    // isUsed has a use
    quir.assign_variable @isUsed : !quir.cbit<1> = %false_cbit
    %use = quir.use_variable @isUsed : !quir.cbit<1>
    %cast = "quir.cast"(%use) : (!quir.cbit<1>) -> i1
    memref.store %cast, %ref[%ind] : memref<1xi1>

    // isOutput doesn't have a use, but is output
    quir.assign_variable @isOutput : !quir.cbit<1> = %false_cbit

    // storeOnly no uses, only assignment/store
    // UNUSED-NOT: quir.assign_variable @storeOnly : !quir.cbit<1> =
    quir.assign_variable @storeOnly : !quir.cbit<1> = %false_cbit

    // notUsed has a useOp, but the result Value is use_empty()
    // UNUSED-NOT: quir.use_variable @notUsed : !quir.cbit<1>
    %notUsed = quir.use_variable @notUsed : !quir.cbit<1>

    return
}
