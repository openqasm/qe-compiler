//===- CLIConfig.h - CLI Configuration builder ------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023, 2024.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  Populate the configuration from the CLI.
///
//===----------------------------------------------------------------------===//
#ifndef QSSC_CLICONFIG_H
#define QSSC_CLICONFIG_H

#include "Config/QSSConfig.h"

#include "llvm/Support/CommandLine.h"

namespace qssc::config {

/// @brief Get the CLI category for the QSS compiler.
/// @return The reference to the CLI category for the compiler.
llvm::cl::OptionCategory &getQSSCCLCategory();

/// @brief Get the CLI category for the QSS compiler mlir-opt options.
/// @return The reference to the CLI category for the compiler.
llvm::cl::OptionCategory &getQSSOptCLCategory();

/// @brief Build a QSSConfig from input CLI arguments.
///
/// When the compiler is invoked it loads the CLI
/// using the MLIR/LLVM CLI library. This enables the
/// inheritance of all of MLIR's powerful CLI functionality.
///
/// The qss-compiler adds several cli arguments to
/// configure the QSSConfig through the CLIConfigBuilder.
class CLIConfigBuilder : public QSSConfigBuilder {
public:
  explicit CLIConfigBuilder();
  static void registerCLOptions(mlir::DialectRegistry &registry);
  llvm::Error populateConfig(QSSConfig &config) override;
};

} // namespace qssc::config
#endif // QSS_CLICONFIG_H
