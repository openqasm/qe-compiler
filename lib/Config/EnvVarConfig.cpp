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

using namespace qssc::config;


llvm::Error EnvVarConfigBuilder::populateConfig(QSSConfig &config) {
  if (auto err = populateConfigurationPath_(config))
    return err;

  if (auto err = populateTarget_(config))
    return err;

  return llvm::Error::success();
}

llvm::Error EnvVarConfigBuilder::populateConfigurationPath_(QSSConfig &config) {
  if (configurationPath != "")
    config.targetConfigPath = configurationPath;
  return llvm::Error::success();
}

llvm::Error EnvVarConfigBuilder::populateTarget_(QSSConfig &config) {
  if (targetStr != "")
    config.targetName = targetStr;
  return llvm::Error::success();
}
