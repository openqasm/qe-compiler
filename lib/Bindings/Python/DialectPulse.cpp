//===- DialectPulse.cpp -  -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pulse.h"

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace llvm;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::adaptors;

void populateDialectPulseSubmodule(const pybind11::module &m) {
  //===-------------------------------------------------------------------===//
  // CaptureType
  //===-------------------------------------------------------------------===//

  auto captureType =
      mlir_type_subclass(m, "CaptureType", pulseTypeIsACaptureType);
  captureType.def_classmethod(
      "get",
      [](py::object cls, MlirContext ctx) {
        return cls(pulseCaptureTypeGet(ctx));
      },
      "Get an instance of CaptureType in given context.", py::arg("cls"),
      py::arg("context") = py::none());

  //===-------------------------------------------------------------------===//
  // FrameType
  //===-------------------------------------------------------------------===//

  auto frameType = mlir_type_subclass(m, "FrameType", pulseTypeIsAFrameType);
  frameType.def_classmethod(
      "get",
      [](py::object cls, MlirContext ctx) {
        return cls(pulseFrameTypeGet(ctx));
      },
      "Get an instance of FrameType in given context.", py::arg("cls"),
      py::arg("context") = py::none());

  //===-------------------------------------------------------------------===//
  // KernelType
  //===-------------------------------------------------------------------===//

  auto kernelType = mlir_type_subclass(m, "KernelType", pulseTypeIsAKernelType);
  kernelType.def_classmethod(
      "get",
      [](py::object cls, MlirContext ctx) {
        return cls(pulseKernelTypeGet(ctx));
      },
      "Get an instance of KernelType in given context.", py::arg("cls"),
      py::arg("context") = py::none());

  //===-------------------------------------------------------------------===//
  // MixedFrameType
  //===-------------------------------------------------------------------===//

  auto mixedFrameType =
      mlir_type_subclass(m, "MixedFrameType", pulseTypeIsAMixedFrameType);
  mixedFrameType.def_classmethod(
      "get",
      [](py::object cls, MlirContext ctx) {
        return cls(pulseMixedFrameTypeGet(ctx));
      },
      "Get an instance of MixedFrameType in given context.", py::arg("cls"),
      py::arg("context") = py::none());


  //===-------------------------------------------------------------------===//
  // PortType
  //===-------------------------------------------------------------------===//

  auto portType = mlir_type_subclass(m, "PortType", pulseTypeIsAPortType);
  portType.def_classmethod(
      "get",
      [](py::object cls, MlirContext ctx) {
        return cls(pulsePortTypeGet(ctx));
      },
      "Get an instance of PortType in given context.", py::arg("cls"),
      py::arg("context") = py::none());

  //===-------------------------------------------------------------------===//
  // WaveformType
  //===-------------------------------------------------------------------===//

  auto waveformType =
      mlir_type_subclass(m, "WaveformType", pulseTypeIsAWaveformType);
  waveformType.def_classmethod(
      "get",
      [](py::object cls, MlirContext ctx) {
        return cls(pulseWaveformTypeGet(ctx));
      },
      "Get an instance of WaveformType in given context.", py::arg("cls"),
      py::arg("context") = py::none());
}

PYBIND11_MODULE(_ibmDialectsPulse, m) {
  m.doc() = "IBM Quantum Pulse dialect.";
  populateDialectPulseSubmodule(m);

  //===--------------------------------------------------------------------===//
  // Pulse dialect
  //===--------------------------------------------------------------------===//
  auto pulse_m = m.def_submodule("pulse");

  pulse_m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__pulse__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load)
           mlirDialectHandleLoadDialect(handle, context);
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}
