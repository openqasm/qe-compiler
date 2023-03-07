//===- lib.cpp --------------------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "llvm/ADT/Optional.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#define DEBUG_TYPE "PyQSSC"

/// Call into the qss-compiler via an interface to qss-compile's command line
/// argument.
pybind11::tuple py_compile_by_args(const std::vector<std::string> &args,
                                   bool outputAsStr,
                                   qssc::DiagnosticCallback onDiagnostic) {
  std::string outputStr("");

  LLVM_DEBUG(
    std::cout << "params passed from python to C++:\n";
    for (auto &str : args)
      std::cout << str << std::endl;
  );

  // TODO: need a C++ interface into the compiler with fewer detours. the python
  // api (inspired by IREE's python bindings) can be a start.
  std::vector<char const *> argv;
  argv.reserve(args.size() + 1);
  for (auto &str : args)
    argv.push_back(str.c_str());
  argv.push_back(nullptr);

  int status = qssc::compile(args.size(), argv.data(),
                             outputAsStr ? &outputStr : nullptr,
                             std::move(onDiagnostic));
  bool success = status == 0;

  return pybind11::make_tuple(success, pybind11::bytes(outputStr));
}

pybind11::tuple
py_link_file(const std::string &inputPath, const std::string &outputPath,
             const std::string &target,
             const std::unordered_map<std::string, double> &parameters) {

  LLVM_DEBUG(
    std::cout << "input " << inputPath << "\n";
    std::cout << "output " << outputPath << "\n";
    std::cout << "parameters (as seen from C++): \n";

    for (auto &item : parameters)
      std::cout << item.first << " = " << item.second << "\n";
  );

  auto successOrErr =
      qssc::bindParameters(target, inputPath, outputPath, parameters);

  if (successOrErr) {
    std::string errorMsg;
    llvm::raw_string_ostream errorMsgStream(errorMsg);
    llvm::logAllUnhandledErrors(std::move(successOrErr), errorMsgStream,
                                "Error: ");

    return pybind11::make_tuple(false, errorMsg);
  }
  return pybind11::make_tuple(true, pybind11::none());
}

PYBIND11_MODULE(py_qssc, m) {

  m.doc() = "Python bindings for the QSS Compiler.";

  m.def("_compile_with_args", &py_compile_by_args, pybind11::call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>(), "Call compiler via cli qss-compile");
  m.def("_link_file", &py_link_file, pybind11::call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>(), "Call the linker tool");

  pybind11::enum_<qssc::ErrorCategory>(m, "ErrorCategory",
                                       pybind11::arithmetic())
      .value("OpenQASM3ParseFailure",
             qssc::ErrorCategory::OpenQASM3ParseFailure)
      .value("UncategorizedError", qssc::ErrorCategory::UncategorizedError)
      .export_values();

  pybind11::enum_<qssc::Severity>(m, "Severity")
      .value("Info", qssc::Severity::Info)
      .value("Warning", qssc::Severity::Warning)
      .value("Error", qssc::Severity::Error)
      .value("Fatal", qssc::Severity::Fatal)
      .export_values();

  pybind11::class_<qssc::Diagnostic>(m, "Diagnostic")
      .def_readonly("severity", &qssc::Diagnostic::severity)
      .def_readonly("category", &qssc::Diagnostic::category)
      .def_readonly("message", &qssc::Diagnostic::message)
      .def("__str__", &qssc::Diagnostic::toString)
      .def(pybind11::pickle(
          [](const qssc::Diagnostic &d) {
            // __getstate__ serializes the C++ object into a tuple
            return pybind11::make_tuple(d.severity, d.category, d.message);
          },
          [](pybind11::tuple const &t) {
            // __setstate__ restores the C++ object from a tuple
            if (t.size() != 3)
              throw std::runtime_error("invalid state for unpickling");

            auto severity = t[0].cast<qssc::Severity>();
            auto category = t[1].cast<qssc::ErrorCategory>();
            auto message = t[2].cast<std::string>();

            return qssc::Diagnostic(severity, category, std::move(message));
          }));
}
