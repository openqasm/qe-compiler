//===- DialectOQ3.cpp - OQ3 Dialect python bindings -------------*- C++ -*-===//
//
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
//
//===----------------------------------------------------------------------===//
///
///  This file implements the python bindings for the OQ3 dialect
///
//===----------------------------------------------------------------------===//

#include "qss-c/Dialect/OQ3.h"

#include "mlir-c/IR.h"
// NOLINTNEXTLINE(misc-include-cleaner)
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include "pybind11/cast.h"
#include "pybind11/detail/common.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"

namespace py = pybind11;
using namespace llvm;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::adaptors;

void populateDialectOQ3Submodule(const pybind11::module &m) {}

PYBIND11_MODULE(_ibmDialectsOQ3, m) {
  m.doc() = "IBM Quantum OQ3 dialect.";
  populateDialectOQ3Submodule(m);

  //===--------------------------------------------------------------------===//
  // OQ3 dialect
  //===--------------------------------------------------------------------===//
  auto quir_m = m.def_submodule("oq3");

  quir_m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        const MlirDialectHandle handle = mlirGetDialectHandle__oq3__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load)
          mlirDialectHandleLoadDialect(handle, context);
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}
