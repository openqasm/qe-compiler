// RUN: qss-compiler -X=mlir --remove-unused-variables %s | FileCheck %s --check-prefix=UNUSED

// UNUSED: oq3.variable_decl @isUsed : !quir.cbit<1>
// UNUSED: oq3.variable_decl {output} @isOutput : !quir.cbit<1>
// UNUSED-NOT: oq3.variable_decl @storeOnly : !quir.cbit<1>
// UNUSED-NOT: oq3.variable_decl @notUsed : !quir.cbit<1>
oq3.variable_decl @isUsed : !quir.cbit<1>
oq3.variable_decl {output} @isOutput : !quir.cbit<1>
oq3.variable_decl @storeOnly : !quir.cbit<1>
oq3.variable_decl @notUsed : !quir.cbit<1>
// UNUSED: func @variableTests
func @variableTests(%ref : memref<1xi1>, %ind : index) {
    %false = arith.constant false
    %false_cbit = "oq3.cast"(%false) : (i1) -> !quir.cbit<1>

    // isUsed has a use
    oq3.variable_assign @isUsed : !quir.cbit<1> = %false_cbit
    %use = oq3.use_variable @isUsed : !quir.cbit<1>
    %cast = "oq3.cast"(%use) : (!quir.cbit<1>) -> i1
    memref.store %cast, %ref[%ind] : memref<1xi1>

    // isOutput doesn't have a use, but is output
    oq3.variable_assign @isOutput : !quir.cbit<1> = %false_cbit

    // storeOnly no uses, only assignment/store
    // UNUSED-NOT: oq3.variable_assign @storeOnly : !quir.cbit<1> =
    oq3.variable_assign @storeOnly : !quir.cbit<1> = %false_cbit

    // notUsed has a useOp, but the result Value is use_empty()
    // UNUSED-NOT: oq3.use_variable @notUsed : !quir.cbit<1>
    %notUsed = oq3.use_variable @notUsed : !quir.cbit<1>

    return
}
