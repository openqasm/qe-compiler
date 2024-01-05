// Redirect stderr (where timing is printed to) to stdout and then stdout to filecheck
// RUN: qss-compiler %s --target mock --config %TEST_CFG --emit=qem --plaintext-payload --mlir-timing --mlir-disable-threading 2>&1 >/dev/null | FileCheck %s
// (C) Copyright IBM 2024.
//
// This code is part of Qiskit.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

// Check timing output of empty payload.
func.func @main () -> i32 {
  %zero = arith.constant 0 : i32
  return %zero : i32
}


// CHECK: build-config
// CHECK: build-target
// CHECK: parse-mlir
// CHECK: command-line-passes
// CHECK: build-qem
// CHECK:   compile-payload
// CHECK:     build-target-pass-managers
// CHECK:     compile-system
// CHECK:       MockSystem
// CHECK:         passes
// CHECK:           Canonicalizer
// CHECK:         emit-to-payload
// CHECK:         children
// CHECK:           MockController
// CHECK:             passes
// CHECK:               qssc::targets::systems::mock::conversion::MockQUIRToStdPass
// CHECK:               Canonicalizer
// CHECK:               LLVMLegalizeForExport
// CHECK:             emit-to-payload
// CHECK:               build-llvm-payload
// CHECK:                  init-llvm
// CHECK:                  translate-to-llvm-mlir-dialect
// CHECK:                  mlir-to-llvm-ir
// CHECK:                  optimize-llvm
// CHECK:                  build-object-file
// CHECK:                  emit-binary
// CHECK:             emit-to-payload-post-children
// CHECK:           MockDrive_0
// CHECK:             passes
// CHECK:             emit-to-payload
// CHECK:             emit-to-payload-post-children
// CHECK:           MockDrive_1
// CHECK:             passes
// CHECK:             emit-to-payload
// CHECK:             emit-to-payload-post-children
// CHECK:           MockAcquire_0
// CHECK:             passes
// CHECK:             emit-to-payload
// CHECK:             emit-to-payload-post-children
// CHECK:         emit-to-payload-post-children
// CHECK:   write-payload
// CHECK: Rest
// CHECK: Total
