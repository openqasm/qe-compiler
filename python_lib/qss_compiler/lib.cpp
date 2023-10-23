//===- exceptions_enum.cpp --------------------------------------*- C++ -*-===//
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
//
//  This file binds the error categories from C++ to Python
//
//===----------------------------------------------------------------------===//

#include "API/api.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

// Forward the declaration of the functions
py::tuple py_compile_by_args(const std::vector<std::string> &,
                             bool,
                             qssc::DiagnosticCallback);

py::tuple
py_link_file(const std::string &, const bool,
             const std::string &,
             const std::string &, const std::string &,
             const std::unordered_map<std::string, double> &,
             bool,
             qssc::DiagnosticCallback);

// Pybind module
PYBIND11_MODULE(py_qssc, m) {
  m.doc() = "Python bindings for the QSS Compiler.";

  m.def("_compile_with_args", &py_compile_by_args,
        "Call compiler via cli qss-compile");
  m.def("_link_file", &py_link_file, "Call the linker tool");

  py::enum_<qssc::ErrorCategory>(m, "ErrorCategory",
                                       py::arithmetic())
      .value("OpenQASM3ParseFailure",
             qssc::ErrorCategory::OpenQASM3ParseFailure)
      .value("QSSCompilerError", qssc::ErrorCategory::QSSCompilerError)
      .value("QSSCompilerNoInputError", qssc::ErrorCategory::QSSCompilerNoInputError)
      .value("QSSCompilerCommunicationFailure", qssc::ErrorCategory::QSSCompilerCommunicationFailure)
      .value("QSSCompilerEOFFailure", qssc::ErrorCategory::QSSCompilerEOFFailure)
      .value("QSSCompilerNonZeroStatus", qssc::ErrorCategory::QSSCompilerNonZeroStatus)
      .value("QSSCompilationFailure", qssc::ErrorCategory::QSSCompilationFailure)
      .value("QSSLinkerNotImplemented", qssc::ErrorCategory::QSSLinkerNotImplemented)
      .value("QSSLinkSignatureWarning", qssc::ErrorCategory::QSSLinkSignatureWarning)
      .value("QSSLinkSignatureError", qssc::ErrorCategory::QSSLinkSignatureError)
      .value("QSSLinkAddressError", qssc::ErrorCategory::QSSLinkAddressError)
      .value("QSSLinkSignatureNotFound", qssc::ErrorCategory::QSSLinkSignatureNotFound)
      .value("QSSLinkArgumentNotFoundWarning", qssc::ErrorCategory::QSSLinkArgumentNotFoundWarning)
      .value("QSSLinkInvalidPatchTypeError", qssc::ErrorCategory::QSSLinkInvalidPatchTypeError)
      .value("UncategorizedError", qssc::ErrorCategory::UncategorizedError)
      .export_values();

  py::enum_<qssc::Severity>(m, "Severity")
      .value("Info", qssc::Severity::Info)
      .value("Warning", qssc::Severity::Warning)
      .value("Error", qssc::Severity::Error)
      .value("Fatal", qssc::Severity::Fatal)
      .export_values();

  py::class_<qssc::Diagnostic>(m, "Diagnostic")
      .def_readonly("severity", &qssc::Diagnostic::severity)
      .def_readonly("category", &qssc::Diagnostic::category)
      .def_readonly("message", &qssc::Diagnostic::message)
      .def("__str__", &qssc::Diagnostic::toString)
      .def(py::pickle(
          [](const qssc::Diagnostic &d) {
            // __getstate__ serializes the C++ object into a tuple
            return py::make_tuple(d.severity, d.category, d.message);
          },
          [](py::tuple const &t) {
            // __setstate__ restores the C++ object from a tuple
            if (t.size() != 3)
              throw std::runtime_error("invalid state for unpickling");

            auto severity = t[0].cast<qssc::Severity>();
            auto category = t[1].cast<qssc::ErrorCategory>();
            auto message = t[2].cast<std::string>();

            return qssc::Diagnostic(severity, category, std::move(message));
          }));
}

