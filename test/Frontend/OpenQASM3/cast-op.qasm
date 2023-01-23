OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

qubit $0;
bit result;
// MLIR: oq3.declare_variable @result : !quir.cbit<1>
// MLIR: %[[MEASURE_RESULT:.*]] = quir.measure({{.*}} -> i1
// MLIR quir.assign_cbit_bit @result<1> [0] : i1 =  %[[MEASURE_RESULT]]
result = measure $0;

// MLIR: %[[RESULT:.*]] = quir.use_variable @result : !quir.cbit<1>
// MLIR: %[[COND:.*]] = "quir.cast"(%[[RESULT]]) : (!quir.cbit<1>) -> i1
// MLIR: scf.if %[[COND]] {
if (result) {
    U(3.1415926, 0, 3.1415926) $0;
}

bit foo = "0";

// TODO: adopt fix in qss-qasm and then add a test case for bool(foo) (i.e.,
//       cast an identifier)
bool b;
// COM: AST-PRETTY: CastNode(from=ASTTypeIdentifier, to=ASTTypeBool, expression=IdentifierNode(name=foo, bits=1))
// b = bool(foo);

// MLIR-DAG: %[[FOO:.*]] = quir.use_variable @foo : !quir.cbit<1>
// MLIR-DAG: %[[RESULT:.*]] = quir.use_variable @result : !quir.cbit<1>
// MLIR: %[[OR:.*]] = quir.cbit_or %[[FOO]], %[[RESULT]] : !quir.cbit<1>
// MLIR: %[[RES2:.*]] = "quir.cast"(%[[OR]]) : (!quir.cbit<1>) -> i1
// MLIR: scf.if %[[RES2]] {
// AST-PRETTY: condition=CastNode(from=ASTTypeBinaryOp, to=ASTTypeBool, expression=BinaryOpNode(type=ASTOpTypeBitOr, left=IdentifierNode(name=foo, bits=1), right=IdentifierNode(name=result, bits=1))
if (bool(foo | result)) {
    foo = result;
}

// MLIR: %[[FOO:.*]] = quir.use_variable @foo : !quir.cbit<1>
// MLIR: %[[RES:.*]] = "quir.cast"(%[[FOO]]) : (!quir.cbit<1>) -> i1
// MLIR: scf.if %[[RES]] {
// AST-PRETTY: condition=CastNode(from=ASTTypeIdentifier, to=ASTTypeBool, expression=IdentifierNode(name=foo, bits=1)
if (bool(foo)) {
   foo = result;
}
