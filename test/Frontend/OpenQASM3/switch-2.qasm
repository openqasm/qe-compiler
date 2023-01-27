OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

int i = 15;
int j = 1;
int k = 2;

float[64] d;

bit c1;

qubit[8] $0;

// MLIR: quir.switch %{{.*}}{
// MLIR:     %angle = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
// MLIR:     %angle_0 = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
// MLIR:     %angle_1 = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
// MLIR:     quir.builtin_U %{{.*}}, %angle, %angle_0, %angle_1 : !quir.qubit<8>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// MLIR: }[1 : {
// MLIR:     %{{.*}} = oq3.use_variable @k : i32
// MLIR:     oq3.assign_variable @j : i32 = %{{.*}}
// MLIR: }2 : {
// MLIR:     %{{.*}} = oq3.use_variable @k : i32
// MLIR:     %{{.*}} = "oq3.cast"(%{{.*}}) : (i32) -> f64
// MLIR:     oq3.assign_variable @d : f64 = %{{.*}}
// MLIR: }3 : {
// MLIR:     %angle = quir.constant #quir.angle<0.000000e+00 : !quir.angle<64>>
// MLIR:     %angle_0 = quir.constant #quir.angle<1.000000e-01 : !quir.angle<64>>
// MLIR:     %angle_1 = quir.constant #quir.angle<2.000000e-01 : !quir.angle<64>>
// MLIR:     quir.builtin_U %{{.*}}, %angle, %angle_0, %angle_1 : !quir.qubit<8>, !quir.angle<64>, !quir.angle<64>, !quir.angle<64>
// MLIR: }]
// AST-PRETTY: SwitchStatementNode(SwitchQuantity(name=i, type=ASTTypeIdentifier),
switch (i) {
    // AST-PRETTY: statements=[
    // AST-PRETTY: CaseStatementNode(case=1, BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=j, bits=32), right=IdentifierNode(name=k, bits=32))
    case 1: {
        j = k;
    }
    break;
    // AST-PRETTY: CaseStatementNode(case=2, BinaryOpNode(type=ASTOpTypeAssign, left=IdentifierNode(name=d, bits=64), right=IdentifierNode(name=k, bits=32))
    case 2: {
        d = k;
    }
    break;
    // AST-PRETTY: CaseStatementNode(case=3, UGateOpNode(params=[AngleNode(value=0.0, bits=64), AngleNode(value=0.1, bits=64), AngleNode(value=0.2, bits=64)], qubits=[], qcparams=[$0])
    case 3: {
        U(0.0, 0.1, 0.2) $0;
    }
    break;
    // AST-PRETTY: ],
    // AST-PRETTY: default statement=[
    // AST-PRETTY: UGateOpNode(params=[AngleNode(value=0.0, bits=64), AngleNode(value=0.0, bits=64), AngleNode(value=0.0, bits=64)], qubits=[], qcparams=[$0])
    // AST-PRETTY: ])
    default: {
        U(0, 0, 0) $0;
    }
    break;
}
