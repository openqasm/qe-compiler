// Ensure bytecode is emitted
// RUN: qss-compiler %s --emit=bytecode | hexdump -C | FileCheck %s --check-prefix BC
// Ensure bytecode is emitted with file extension
// RUN: qss-compiler %s -o bytecode_output.bc && hexdump -C bytecode_output.bc | FileCheck %s --check-prefix BC

// Check bytecode is parse/emit roundtripable
// RUN: qss-compiler %s --emit=bytecode -o test.bc && qss-compiler test.bc -X=bytecode --emit=mlir | FileCheck %s
// Check that the compiler automatically differentiates between MLIR/bytecode
// RUN: qss-compiler %s --emit=bytecode -o test.bc && qss-compiler test.bc -X=mlir --emit=mlir | FileCheck %s

// Look for the bytecode magic number
// https://mlir.llvm.org/docs/BytecodeFormat/#magic-number
// BC: 4d 4c ef 52

// CHECK: module {
func.func @dummy() {
// CHECK: func.func @dummy() {
    return
    // CHECK: return
}
