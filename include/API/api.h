//===- api.h ----------------------------------------------------*- C++ -*-===//
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

#ifndef QSS_COMPILER_LIB_H
#define QSS_COMPILER_LIB_H

#include "API/errors.h"
#include "Config/QSSConfig.h"

#include "Config/QSSConfig.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/Timing.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace qssc {

// The API implementation is based on that of MLIROptMain in the core
// MLIR project. The hope is that this helps standardize CLI tooling and make
// forwards compatiability more straightforward

/// Register and parse command line options.
/// @param argc Commandline argc to parse.
/// @param argv Commandline argv to parse.
/// @param toolName for the header displayed by `--help`.
/// @param registry should contain all the dialects that can be parsed in the
/// source.
void registerAndParseCLIOptions(int argc, const char **argv,
                                llvm::StringRef toolName,
                                mlir::DialectRegistry &registry);

/// Register and parse command line tool options.
/// @param argc Commandline argc to parse.
/// @param argv Commandline argv to parse.
/// @param toolName for the header displayed by `--help`.
/// @param registry should contain all the dialects that can be parsed in the
/// source.
/// @return inputFilename and outputFilename command line option values.
std::pair<std::string, std::string>
registerAndParseCLIToolOptions(int argc, const char **argv,
                               llvm::StringRef toolName,
                               mlir::DialectRegistry &registry);

/// Perform the core processing behind `qss-compiler`
/// @param outputStream to emit to.
/// @param buffer to parse and process.
/// @param registry should contain all the dialects that can be parsed in the
/// source.
/// @param config compilation configuration.
/// @param diagnosticCb callback for error diagnostic processsing.
/// @param timing scope for time tracking
llvm::Error compileMain(llvm::raw_ostream &outputStream,
                        std::unique_ptr<llvm::MemoryBuffer> buffer,
                        mlir::DialectRegistry &registry,
                        const qssc::config::QSSConfig &config,
                        OptDiagnosticCallback diagnosticCb,
                        mlir::TimingScope &timing);

/// Implementation for tools like `qss-compiler`.
/// @param argc Commandline argc to parse.
/// @param argv Commandline argv to parse.
/// @param inputFilename input filename to parse.
/// @param outputFilename output filename to emit to.
/// @param registry should contain all the dialects that can be parsed in the
/// source.
/// @param diagnosticCb callback for error diagnostic processsing.
llvm::Error compileMain(int argc, const char **argv,
                        llvm::StringRef inputFilename,
                        llvm::StringRef outputFilename,
                        mlir::DialectRegistry &registry,
                        OptDiagnosticCallback diagnosticCb);

/// Implementation for tools like `qss-compiler`.
/// @param argc Commandline argc to parse.
/// @param argv Commandline argv to parse.
/// @param registry should contain all the dialects that can be parsed in the
/// source.
/// @param diagnosticCb callback for error diagnostic processsing.
llvm::Error compileMain(int argc, const char **argv, llvm::StringRef toolName,
                        mlir::DialectRegistry &registry,
                        OptDiagnosticCallback diagnosticCb);

/// Implementation for tools like `qss-compiler` with provided registry with
/// default project dialects loaded
/// @param argc Commandline argc to parse.
/// @param argv Commandline argv to parse.
/// @param diagnosticCb callback for error diagnostic processsing.
llvm::Error compileMain(int argc, const char **argv, llvm::StringRef toolName,
                        OptDiagnosticCallback diagnosticCb);

/// Helper wrapper to return the result of compileMain directly from main
inline int asMainReturnCode(llvm::Error err) {
  return err ? EXIT_FAILURE : EXIT_SUCCESS;
}

/// @brief Call the parameter binder
/// @param target name of the target to employ
/// @param action name of the emit action of input and output
/// @param moduleInputPath path of the module to use as input
/// @param payloadOutputPath path of the payload to generate as output
/// @param arguments bindings for the parameters in the module to apply
/// @param treatWarningsAsErrors return errors in place of warnings
/// @param diagnosticCb an optional callback that will receive emitted
/// diagnostics
/// @return 0 on success
int bindArguments(std::string_view target, qssc::config::EmitAction action,
                  std::string_view configPath, std::string_view moduleInput,
                  std::string_view payloadOutputPath,
                  std::unordered_map<std::string, double> const &arguments,
                  bool treatWarningsAsErrors, bool enableInMemoryInput,
                  std::string *inMemoryOutput,
                  const OptDiagnosticCallback &onDiagnostic);

} // namespace qssc
#endif // QSS_COMPILER_LIB_H
