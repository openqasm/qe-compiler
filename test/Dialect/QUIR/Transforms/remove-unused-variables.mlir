// RUN: qss-compiler -X=mlir --remove-unused-variables %s | FileCheck %s --check-prefix=UNUSED

//
// This code is part of Qiskit.
//
// (C) Copyright IBM 2023.
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
