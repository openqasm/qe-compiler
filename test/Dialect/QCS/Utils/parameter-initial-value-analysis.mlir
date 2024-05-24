// RUN: qss-opt --qcs-parameter-initial-value-analysis-print --pass-pipeline='builtin.module(builtin.module(qcs-parameter-initial-value-analysis))' --mlir-disable-threading %s | FileCheck %s --check-prefixes CHECK,NESTED
// RUN: qss-opt --qcs-parameter-initial-value-analysis-print --qcs-parameter-initial-value-analysis %s | FileCheck %s

//
// This code is part of Qiskit.
//
// (C) Copyright IBM 2024.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

module {
    // without nested pass manager should only find
    // alpha and beta
    qcs.parameter_load "alpha" : !quir.angle<64> {initialValue = 1.000000e+00 :f64}
    qcs.parameter_load "beta" : f64 {initialValue = 2.000000e+00 :f64}
    module @first {
        // nested test should find alpha and beta
        qcs.parameter_load "theta" : !quir.angle<64> {initialValue = 3.140000e+00 :f64}
        qcs.parameter_load "phi" : f64 {initialValue = 1.500000e+00 :f64}
    }
    module @second {
        // test module without declare_parameter
        // should find alpha and beta when nested
    }
}

// CHECK-DAG: alpha = 1.000000e+00
// CHECK-DAG: beta =  2.000000e+00
// NESTED-DAG: theta = 3.140000e+00
// NESTED-DAG: phi = 1.500000e+00
