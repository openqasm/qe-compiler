OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --check-prefix MLIR

// MLIR: module
// MLIR-DAG: func @main


bit a;
bit b;
bit c;

// AST-PRETTY-COUNT-2: DeclarationNode(type=ASTTypeBitset
// MLIR-DAG: oq3.declare_variable @a : !quir.cbit<1>
// MLIR-DAG: oq3.declare_variable @b : !quir.cbit<1>
// MLIR-DAG: oq3.declare_variable @c : !quir.cbit<1>

qubit $0;
qubit $1;
qubit $2;

// AST-PRETTY-COUNT-3: DeclarationNode(type=ASTTypeQubitContainer
// MLIR-DAG-COUNT-3: quir.declare_qubit

gate x q0 {
 U(3.14159265359, 0.0, 3.14159265359) q0;
}

x $0;


a = measure $0; // expected "1"
b = measure $1; // expected "0"
c = measure $2; // expected "1"

bit meas_and;
// MLIR-DAG: oq3.declare_variable @meas_and : !quir.cbit<1>

// MLIR-DAG: [[A:%.*]] = oq3.variable_load @a
// MLIR-DAG: [[B:%.*]] = oq3.variable_load @b
// MLIR-DAG: [[C:%.*]] = oq3.variable_load @c
// MLIR-DAG: [[A_OR_C:%.*]] = oq3.cbit_or [[A]], [[C]] : !quir.cbit<1>
// MLIR-DAG: [[A_OR_C__AND_B:%.*]] = oq3.cbit_and [[A_OR_C]], [[B]]

// AST-PRETTY: condition=BinaryOpNode(type=ASTOpTypeCompEq, left=BinaryOpNode(type=ASTOpTypeBitAnd, left=BinaryOpNode(type=ASTOpTypeBitOr, left=IdentifierNode(name=a, bits=1), right=IdentifierNode(name=c, bits=1))
// AST-PRETTY: , right=IdentifierNode(name=b, bits=1))
// AST-PRETTY: , right=IntNode(signed=true, value=1, bits=32))
if (((a | c) & b) == 1) {
    meas_and = measure $0;
} else {
    meas_and = measure $1;
}
// MLIR: oq3.cbit_assign_bit @meas_and<1> [0] : i1 =
// on hardware, expect meas_and to become 0

bit d;

if (bool(a | b)) {
// AST-PRETTY: condition=CastNode(from=ASTTypeBinaryOp, to=ASTTypeBool, expression=BinaryOpNode(type=ASTOpTypeBitOr, left=IdentifierNode(name=a, bits=1), right=IdentifierNode(name=b, bits=1))
// MLIR-DAG: [[A:%.*]] = oq3.variable_load @a
// MLIR-DAG: [[B:%.*]] = oq3.variable_load @b
// MLIR: [[CBIT_OR_RES:%[0-9]+]] = oq3.cbit_or [[A]], [[B]]
// MLIR: "oq3.cast"([[CBIT_OR_RES]]) {{.*}} -> i1
    d = measure $0;
} else {
    d = measure $1;
}
// MLIR: oq3.cbit_assign_bit @d<1> [0] : i1 =
// on hardware, expect d to be 1

bit e;

if (bool(a ^ b))  {
// AST-PRETTY: BinaryOpNode(type=ASTOpTypeXor, left=IdentifierNode(name=a, bits=1), right=IdentifierNode(name=b, bits=1))
// MLIR-DAG: [[A:%.*]] = oq3.variable_load @a
// MLIR-DAG: [[B:%.*]] = oq3.variable_load @b
// MLIR: [[CBIT_XOR_RES:%[0-9]+]] = oq3.cbit_xor [[A]], [[B]]
// MLIR: "oq3.cast"([[CBIT_XOR_RES]]) {{.*}} -> i1
    e = measure $0;
} else {
    e = measure $1;
}
// MLIR: oq3.cbit_assign_bit @e<1> [0] : i1 =
// on hardware, expect e to be 1

bit f = "0";

f = e | d;

// MLIR: [[F:%.*]] = oq3.variable_load @f : !quir.cbit<1>
// MLIR: [[BOOL_F:%.*]] = "oq3.cast"([[F]]) : (!quir.cbit<1>) -> i1
// MLIR: [[TRUE:%.*]] = arith.constant true
// MLIR: [[NOT:%.*]] = arith.cmpi ne, [[BOOL_F]], [[TRUE]] : i1
// MLIR: [[NOT_CBIT:%.*]] = "oq3.cast"([[NOT]]) : (i1) -> !quir.cbit<1>
// MLIR: oq3.variable_assign @f : !quir.cbit<1> = [[NOT_CBIT]]
f = !f;
