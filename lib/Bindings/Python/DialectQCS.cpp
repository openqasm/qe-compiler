//===- DialectQCS.cpp - QCS Dialect python bindings -------------*- C++ -*-===//
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
///  This file implements the python bindings for the QCS dialect
///
//===----------------------------------------------------------------------===//

#include "qss-c/Dialect/QCS.h"

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

// Intentionally left blank placeholder and for a common pattern across dialects
void populateDialectQCSSubmodule(const pybind11::module &m) {}

PYBIND11_MODULE(_qeDialectsQCS, m) {
  m.doc() = "IBM Quantum QCS dialect.";
  populateDialectQCSSubmodule(m);

  //===--------------------------------------------------------------------===//
  // QCS dialect
  //===--------------------------------------------------------------------===//
  auto qcs_m = m.def_submodule("qcs");

  qcs_m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        const MlirDialectHandle handle = mlirGetDialectHandle__qcs__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load)
          mlirDialectHandleLoadDialect(handle, context);
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}
