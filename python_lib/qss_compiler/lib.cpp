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

#include "Config/QSSConfig.h"
#include "errors.h"
#include "lib_enums.h"

#include "API/api.h"
// <<<<<<< HEAD
// #include <Config/QSSConfig.h>
// =======
#include "Dialect/RegisterDialects.h"
#include "Dialect/RegisterPasses.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <iostream>
#include <memory>
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

namespace {

std::vector<const char *> buildArgv(std::vector<std::string> &args) {
#ifndef NDEBUG
  std::cout << "params passed from python to C++:\n";
  for (auto &str : args)
    std::cout << str << '\n';
#endif

  // TODO: need a C++ interface into the compiler with fewer detours. the python
  // api (inspired by IREE's python bindings) can be a start.
  std::vector<const char *> argv;
  argv.reserve(args.size());
  for (auto &str : args)
    argv.push_back(str.c_str());

  return argv;
}

llvm::Expected<mlir::DialectRegistry> buildRegistry() {
  // Register the standard passes with MLIR.
  // Must precede the command line parsing.
  if (auto err = qssc::dialect::registerPasses())
    return std::move(err);

  mlir::DialectRegistry registry;

  // Add the following to include *all* QSS core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  qssc::dialect::registerDialects(registry);
  return std::move(registry);
}

llvm::Error compile(llvm::raw_ostream &outputStream,
                    std::unique_ptr<llvm::MemoryBuffer> input,
                    std::vector<std::string> &args,
                    qssc::DiagnosticCallback onDiagnostic,
                    llvm::StringRef outputPath = "-") {

  auto argv = buildArgv(args);

  auto registry = buildRegistry();
  if (auto err = registry.takeError())
    return err;

  /// TODO: We should not be performing argument parsing in the Python API.
  qssc::registerAndParseCLIOptions(argv.size(), argv.data(), "pyqssc\n",
                                   *registry);

  mlir::DefaultTimingManager tm;
  mlir::applyDefaultTimingManagerCLOptions(tm);
  mlir::TimingScope timing = tm.getRootScope();

  mlir::TimingScope buildConfigTiming = timing.nest("build-config");
  llvm::StringRef inputPath = "-";
  auto bufferIdentifier = input->getBufferIdentifier();
  if (bufferIdentifier != "<stdin>")
    inputPath = bufferIdentifier;
  auto configResult = qssc::config::buildToolConfig(inputPath, outputPath);

  if (auto err = configResult.takeError())
    return err;
  qssc::config::QSSConfig const config = configResult.get();
  buildConfigTiming.stop();

  if (auto err = qssc::compileMain(outputStream, std::move(input), *registry,
                                   config, std::move(onDiagnostic), timing))
    return err;

  return llvm::Error::success();
}

py::tuple compileOptionalOutput(std::optional<std::string> outputFile,
                                std::unique_ptr<llvm::MemoryBuffer> input,
                                std::vector<std::string> &args,
                                qssc::DiagnosticCallback onDiagnostic) {
  bool success = true;
  if (outputFile.has_value()) {
    std::string errorMessage;
    auto output = mlir::openOutputFile(outputFile.value(), &errorMessage);
    if (!output) {
      llvm::errs() << "Failed to open output file: " << errorMessage;
      return py::make_tuple(false, py::bytes(""));
    }
    if (auto err = compile(output->os(), std::move(input), args,
                           std::move(onDiagnostic), outputFile.value()))
      success = false;

    if (success)
      output->keep();

    return py::make_tuple(success, py::bytes(""));
  }

  std::string outputString;
  // NOLINTNEXTLINE(misc-const-correctness)
  llvm::raw_string_ostream output(outputString);
  if (auto err =
          compile(output, std::move(input), args, std::move(onDiagnostic)))
    success = false;

  return py::make_tuple(success, py::bytes(outputString));
}

} // anonymous namespace

/// Call into the qss-compiler to compile input bytes
py::tuple py_compile_bytes(const std::string &bytes,
                           const std::optional<std::string> &outputFile,
                           std::vector<std::string> &args,
                           qssc::DiagnosticCallback onDiagnostic) {

  // Set up the input file.
  std::unique_ptr<llvm::MemoryBuffer> input =
      llvm::MemoryBuffer::getMemBuffer(bytes);

  return compileOptionalOutput(outputFile, std::move(input), args,
                               std::move(onDiagnostic));
}

/// Call into the qss-compiler to compile input file
py::tuple py_compile_file(const std::string &inputFile,
                          const std::optional<std::string> &outputFile,
                          std::vector<std::string> &args,
                          qssc::DiagnosticCallback onDiagnostic) {
  // Set up the input file.
  std::string errorMessage;
  auto input = mlir::openInputFile(inputFile, &errorMessage);
  if (!input) {
    llvm::errs() << "Failed to open input file: " << errorMessage;
    return py::make_tuple(false, py::bytes(""));
  }

  return compileOptionalOutput(outputFile, std::move(input), args,
                               std::move(onDiagnostic));
}

py::tuple py_link_file(const std::string &input, const bool enableInMemoryInput,
                       const std::string &outputPath, const std::string &target,
                       const std::string &configPath,
                       const std::unordered_map<std::string, double> &arguments,
                       bool treatWarningsAsErrors,
                       qssc::DiagnosticCallback onDiagnostic) {

  std::string inMemoryOutput("");

  int const status = qssc::bindArguments(
      target, qssc::config::EmitAction::QEM, configPath, input, outputPath,
      arguments, treatWarningsAsErrors, enableInMemoryInput, &inMemoryOutput,
      std::move(onDiagnostic));

  bool const success = status == 0;
#ifndef NDEBUG
  std::cerr << "Link " << (success ? "successful" : "failed") << '\n';
#endif
  return py::make_tuple(success, py::bytes(inMemoryOutput));
}

// Pybind module
PYBIND11_MODULE(py_qssc, m) {
  m.doc() = "Python bindings for the QSS Compiler.";

  m.def("_compile_bytes", &py_compile_bytes,
        "Call qss-compiler to compile input bytes");
  m.def("_compile_file", &py_compile_file,
        "Call qss-compiler to compile input file");
  m.def("_link_file", &py_link_file, "Call the linker tool");

  addErrorCategory(m);
  addSeverity(m);
  addDiagnostic(m);
}
