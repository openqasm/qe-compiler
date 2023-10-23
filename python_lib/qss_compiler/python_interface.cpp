//===- python_interfaces.cpp ------------------------------------*- C++ -*-===//
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
//  This file contains the functions that act as an interface between the
//  python and c++ functionalities.
//
//===----------------------------------------------------------------------===//

#include "API/api.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "llvm/ADT/Optional.h"
#include <llvm/Support/Error.h>

#include <iostream>
#include <string>
#include <unordered_map>
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
    std::cout << str << std::endl;
#endif

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

#ifndef NDEBUG
  std::cerr << "Compile " << (success ? "successful" : "failed") << std::endl;
#endif

  return py::make_tuple(success, py::bytes(outputStr));
}

py::tuple
py_link_file(const std::string &input, const bool enableInMemoryInput,
             const std::string &outputPath,
             const std::string &target, const std::string &configPath,
             const std::unordered_map<std::string, double> &arguments,
             bool treatWarningsAsErrors,
             qssc::DiagnosticCallback onDiagnostic) {

  std::string inMemoryOutput("");

  int status = qssc::bindArguments(target, configPath, input, outputPath, arguments,
                                   treatWarningsAsErrors, enableInMemoryInput,
                                   &inMemoryOutput,
                                   std::move(onDiagnostic));

  bool success = status == 0;
#ifndef NDEBUG
  std::cerr << "Link " << (success ? "successful" : "failed") << std::endl;
#endif
  return py::make_tuple(success, py::bytes(inMemoryOutput));
}
