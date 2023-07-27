//===- DialectPulse.cpp -  -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "oq3.h"

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace llvm;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::adaptors;

void populateDialectOQ3Submodule(const pybind11::module &m) {
  
}

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
        MlirDialectHandle handle = mlirGetDialectHandle__oq3__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load)
          mlirDialectHandleLoadDialect(handle, context);
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}
