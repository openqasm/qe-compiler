//===- CLIConfig.h - CLI Configuration builder ------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
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
llvm::cl::OptionCategory &getQSSCCategory();

/// @brief Build a QSSConfig from input CLI arguments.
///
/// When the compiler is invoked it loads the CLI
/// using the MLIR/LLVM CLI library. This enables the
/// inheritance of all of MLIR's powerful CLI functionality.
///
/// The qss-compiler adds several cli arguments to
/// configure the QSSConfig through the CLIConfigBuilder.
///
/// These currently are:
/// - `--target=<str>`: Sets QSSConfig::targetName.
/// - `--config=<str>`: Sets QSSConfig::targetConfigPath.
/// - `--allow-unregistered-dialect=<bool>`: Sets
/// QSSConfig::allowUnregisteredDialects.
/// - `--add-target-passes=<bool>`: Sets QSSConfig::addTargetPasses.
/// - `--verbosity=<str>` : Sets verbosity level for output
///
class CLIConfigBuilder : public QSSConfigBuilder {
public:
  llvm::Error populateConfig(QSSConfig &config) override;

private:
  llvm::Error populateConfigurationPath_(QSSConfig &config);
  llvm::Error populateTarget_(QSSConfig &config);
  llvm::Error populateAllowUnregisteredDialects_(QSSConfig &config);
  llvm::Error populateAddTargetPasses_(QSSConfig &config);
  llvm::Error populateVerbosity_(QSSConfig &config);
};

} // namespace qssc::config
#endif // QSS_CLICONFIG_H
