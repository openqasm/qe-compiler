//===- DialectPulse.cpp -  -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "qss-c/Dialect/QUIR.h"

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

void populateDialectQUIRSubmodule(const pybind11::module &m) {
  //===-------------------------------------------------------------------===//
  // AngleType
  //===-------------------------------------------------------------------===//

  auto angleType = mlir_type_subclass(m, "AngleType", quirTypeIsAAngleType);
  angleType.def_classmethod(
      "get",
      [](const py::object &cls, unsigned width, MlirContext ctx) {
        return cls(quirAngleTypeGet(ctx, width));
      },
      "Get an instance of AngleType in given context.", py::arg("cls"),
      py::arg("width"), py::arg("context") = py::none());

  //===-------------------------------------------------------------------===//
  // DurationType
  //===-------------------------------------------------------------------===//

  auto durationType =
      mlir_type_subclass(m, "DurationType", quirTypeIsADurationType);
  durationType.def_classmethod(
      "get",
      [](const py::object &cls, MlirContext ctx) {
        return cls(quirDurationTypeGet(ctx));
      },
      "Get an instance of DurationType in given context.", py::arg("cls"),
      py::arg("context") = py::none());
}

PYBIND11_MODULE(_ibmDialectsQUIR, m) {
  m.doc() = "IBM Quantum QUIR dialect.";
  populateDialectQUIRSubmodule(m);

  //===--------------------------------------------------------------------===//
  // QUIR dialect
  //===--------------------------------------------------------------------===//
  auto quir_m = m.def_submodule("quir");

  quir_m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        const MlirDialectHandle handle = mlirGetDialectHandle__quir__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load)
          mlirDialectHandleLoadDialect(handle, context);
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}
