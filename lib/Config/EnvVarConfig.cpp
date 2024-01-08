//===- EnvVarConfigBuilder.cpp - QSSConfig from EnvVars  ----* C++*--------===//
//
// (C) Copyright IBM 2023, 2024.
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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdlib>
#include <cstring>

using namespace qssc::config;

llvm::Error EnvVarConfigBuilder::populateConfig(QSSConfig &config) {
  if (auto err = populateConfigurationPath_(config))
    return err;

  if (auto err = populateTarget_(config))
    return err;

  if (auto err = populateVerbosity_(config))
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

llvm::Error EnvVarConfigBuilder::populateVerbosity_(QSSConfig &config) {
  if (const char *verbosity = std::getenv("QSSC_VERBOSITY")) {
    if (strcmp(verbosity, "ERROR") == 0) {
      config.setVerbosityLevel(QSSVerbosity::Error);
    } else if (strcmp(verbosity, "WARN") == 0) {
      config.setVerbosityLevel(QSSVerbosity::Warn);
    } else if (strcmp(verbosity, "INFO") == 0) {
      config.setVerbosityLevel(QSSVerbosity::Info);
    } else if (strcmp(verbosity, "DEBUG") == 0) {
      config.setVerbosityLevel(QSSVerbosity::Debug);
    } else {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "QSSC_VERBOSITY level unrecognized got (" +
              llvm::StringRef(verbosity) +
              "), options are ERROR, WARN, INFO, or DEBUG\n");
    }
  }
  return llvm::Error::success();
}
