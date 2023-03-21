//===- QUIRTraits.td - QUIR dialect traits -*- C++ -*-========================//
//
// (C) Copyright IBM 2023.
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
//
//===----------------------------------------------------------------------===//
///
///  Traits for the QUIR dialect
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_TRAITS
#define QUIR_TRAITS

include "mlir/IR/OpBase.td"

// QUIR Traits

// Operation applies a quantum channel to the quantum system.
// i.e., it is a completely positive trace perserving (CPTP) map
def CPTPOp : NativeOpTrait<"CPTPOp">{
    let cppNamespace= "::mlir::quir";
}

// Operation applies a unitary transformation to the quantum system.
// Is a subclass of CPTPOp.
def UnitaryOp : NativeOpTrait<"UnitaryOp"> {
   let cppNamespace= "::mlir::quir";
}

#endif // QUIR_TRAITS
