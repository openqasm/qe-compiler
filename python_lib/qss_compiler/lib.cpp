//===- lib.cpp --------------------------------------------------*- C++ -*-===//
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
//  This file generates python bindings for the QSS Compiler API using pybind11
//
//  Developer notes:
//  Pybind11 is a lightweight, easy-to-use package for creating python bindings
//  for C++ code with minimal overhead. It requires C++11 compatible compilers.
//  (Compare against Boost.Python which supports all compilers, but is slower
//  and produces larger binaries.)
//
//  This file is primarily responsible for generating the binding code, which is
//  handled by the PYBIND11_MODULE macro. It also contains the implementation
//  `py_compile`. It's common to separate these, and would be a good idea if the
//  complexity of this file grows.
//
//  The pybind macro accepts two arguments: the module name (`py_qssc`) and a
//  name (`m`) for a new `py::module_` object -- this is the main interface for
//  creating bindings. For instance, `m.def` creates the bindings for a new
//  function.
//
//     Note: `py_qssc` is the name of the shared library being built. Reference
//           it with this name in CMakeLists.
//
//  The first argument to `def` is the name of the new _python_ function, and
//  the second argument is a pointer to the function to bind. We don't point
//  directly to the `compile` function because we want to decouple the function
//  signatures. Besides, the `api.compile` method is optimized for the command
//  line, whereas our function should be pythonic.

//  There is another level of indirection in `qss_compiler/` which is the actual
//  python package! The `_compile` function is used by the user facing
//  `qss_compiler.compile` function. More developer notes can be found in
//  `qss_compiler/__init__.py`.

//  The documentation for pybind11 is quite comprehensive and helpful as a guide
//  to more complicated usage
//  https://pybind11.readthedocs.io/en/stable/
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

