//===- DialectPulse.cpp -  -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "qcs.h"

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace llvm;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::adaptors;

void populateDialectQCSSubmodule(const pybind11::module &m) {
  
}

PYBIND11_MODULE(_ibmDialectsQCS, m) {
  m.doc() = "IBM Quantum QCS dialect.";
  populateDialectQCSSubmodule(m);

  //===--------------------------------------------------------------------===//
  // QCS dialect
  //===--------------------------------------------------------------------===//
  auto quir_m = m.def_submodule("qcs");

  quir_m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__qcs__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load)
          mlirDialectHandleLoadDialect(handle, context);
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}
