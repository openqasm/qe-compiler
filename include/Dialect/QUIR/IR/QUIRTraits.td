//===- QUIRTraits.td - QUIR dialect traits -*- C++ -*-========================//
//
// (C) Copyright IBM 2022, 2023.
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
