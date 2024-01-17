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

#include "errors.h"
#include "lib_enums.h"

#include "API/api.h"

#include <iostream>
#include <optional>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

/// Call into the qss-compiler via an interface to qss-compile's command line
/// argument.
py::tuple py_compile_by_args(const std::vector<std::string> &args,
                             bool outputAsStr,
                             qssc::DiagnosticCallback onDiagnostic) {
  std::string outputStr("");

#ifndef NDEBUG
  std::cout << "params passed from python to C++:\n";
  for (auto &str : args)
    std::cout << str << '\n';
#endif

  // TODO: need a C++ interface into the compiler with fewer detours. the python
  // api (inspired by IREE's python bindings) can be a start.
  std::vector<char const *> argv;
  argv.reserve(args.size() + 1);
  for (auto &str : args)
    argv.push_back(str.c_str());
  argv.push_back(nullptr);

  int const status = qssc::compile(args.size(), argv.data(),
                                   outputAsStr ? &outputStr : nullptr,
                                   std::move(onDiagnostic));
  bool const success = status == 0;

#ifndef NDEBUG
  std::cerr << "Compile " << (success ? "successful" : "failed") << '\n';
#endif

  return py::make_tuple(success, py::bytes(outputStr));
}

py::tuple py_link_file(const std::string &input, const bool enableInMemoryInput,
                       const std::string &outputPath, const std::string &target,
                       const std::string &configPath,
                       const std::unordered_map<std::string, double> &arguments,
                       bool treatWarningsAsErrors,
                       qssc::DiagnosticCallback onDiagnostic) {

  std::string inMemoryOutput("");

  int const status = qssc::bindArguments(
      target, configPath, input, outputPath, arguments, treatWarningsAsErrors,
      enableInMemoryInput, &inMemoryOutput, std::move(onDiagnostic));

  bool const success = status == 0;
#ifndef NDEBUG
  std::cerr << "Link " << (success ? "successful" : "failed") << '\n';
#endif
  return py::make_tuple(success, py::bytes(inMemoryOutput));
}

// Pybind module
PYBIND11_MODULE(py_qssc, m) {
  m.doc() = "Python bindings for the QSS Compiler.";

  m.def("_compile_with_args", &py_compile_by_args,
        "Call compiler via cli qss-compile");
  m.def("_link_file", &py_link_file, "Call the linker tool");

  addErrorCategory(m);
  addSeverity(m);
  addDiagnostic(m);
}
