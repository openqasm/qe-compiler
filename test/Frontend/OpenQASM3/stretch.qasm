OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast-pretty %s | FileCheck %s --match-full-lines --check-prefix AST-PRETTY
// RUN: qss-compiler -X=qasm --emit=mlir %s | FileCheck %s --match-full-lines --check-prefix MLIR

// AST-PRETTY: StretchStatementNode(StretchNode(name=a))
// AST-PRETTY: StretchStatementNode(StretchNode(name=b))
// MLIR: {{.*}} = quir.declare_stretch : !quir.stretch
// MLIR: {{.*}} = quir.declare_stretch : !quir.stretch
stretch a;
stretch b;
