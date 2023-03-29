//===- EnvVarConfig.h - EnvVar Configuration builder ----------*- C++-*----===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  Populate the configuration from environment variables.
///
//===----------------------------------------------------------------------===//
#ifndef QSSC_ENVVARCONFIG_H
#define QSSC_ENVVARCONFIG_H

#include "Config/QSSConfig.h"

namespace qssc::config {

/// @brief Populate arguments of the QSSConfig
/// from environment variables.
///
///
/// The qss-compiler makes several several QSSConfig configuration
/// options configurable from environment variables through the
/// EnvVarConfigBuilder.
///
/// These currently are:
/// - `QSSC_TARGET_NAME`: Sets QSSConfig::targetName.
/// - `QSSC_TARGET_CONFIG_PATH`: Sets QSSConfig::targetConfigPath.
///
class EnvVarConfigBuilder : public QSSConfigBuilder {
public:
  llvm::Error populateConfig(QSSConfig &config) override;

private:
  llvm::Error populateConfigurationPath_(QSSConfig &config);
  llvm::Error populateTarget_(QSSConfig &config);
};

} // namespace qssc::config
#endif // QSS_ENVVARCONFIG_H
