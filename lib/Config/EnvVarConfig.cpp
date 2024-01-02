//===- EnvVarConfigBuilder.cpp - QSSConfig from EnvVars  ----* C++*--------===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  This file implements building the configuration from environemnt variables.
//
//===----------------------------------------------------------------------===//

#include "Config/EnvVarConfig.h"
#include "Config/QSSConfig.h"

#include "llvm/Support/Error.h"

#include <cstdlib>

using namespace qssc::config;

llvm::Error EnvVarConfigBuilder::populateConfig(QSSConfig &config) {
  if (auto err = populateConfigurationPath_(config))
    return err;

  if (auto err = populateTarget_(config))
    return err;

  return llvm::Error::success();
}

llvm::Error EnvVarConfigBuilder::populateConfigurationPath_(QSSConfig &config) {
  if (const char *configurationPath = std::getenv("QSSC_TARGET_CONFIG_PATH"))
    config.targetConfigPath = configurationPath;
  return llvm::Error::success();
}

llvm::Error EnvVarConfigBuilder::populateTarget_(QSSConfig &config) {
  if (const char *targetStr = std::getenv("QSSC_TARGET_NAME"))
    config.targetName = targetStr;
  return llvm::Error::success();
}
