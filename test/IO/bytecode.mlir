// Ensure bytecode is emitted
// RUN: qss-compiler %s --emit=bytecode | xxd -p | FileCheck %s --check-prefix BC

// Check bytecode is parse/emit roundtripable
// RUN: qss-compiler %s --emit=bytecode | qss-compiler %s -X=bytecode --emit=mlir  | FileCheck %s

// Look for the bytecode magic number
// https://mlir.llvm.org/docs/BytecodeFormat/#magic-number
// BC: 4d4cef52

// CHECK: module {
func.func @dummy() {
// CHECK: func.func @dummy() {
    return
    // CHECK: return
}
