//===- CLIConfigBuilder.cpp - QSSConfig from the CLI ------*- C++ -*-------===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file implements building the configuration from the CLI.
///
//===----------------------------------------------------------------------===//

#include "Config/CLIConfig.h"
#include "Config/QSSConfig.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"

#include <string>

using namespace qssc::config;

namespace {
// The space below at the front of the string causes this category to be printed
// first
llvm::cl::OptionCategory
    qsscCat_(" QSS Compiler Options",
             "Options that control high-level behavior of QSS Compiler");

llvm::cl::opt<std::string> configurationPath(
    "config",
    llvm::cl::desc("Path to configuration file or directory (depends on the "
                   "target), - means use the config service"),
    llvm::cl::value_desc("path"), llvm::cl::cat(qsscCat_), llvm::cl::Optional);

llvm::cl::opt<std::string>
    targetStr("target",
              llvm::cl::desc(
                  "Target architecture. Required for machine code generation."),
              llvm::cl::value_desc("targetName"),
              llvm::cl::cat(getQSSCCategory()));

llvm::cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    llvm::cl::desc("Allow operation with no registered dialects"),
    llvm::cl::init(false), llvm::cl::cat(getQSSCCategory()));

llvm::cl::opt<bool> addTargetPasses(
    "add-target-passes", llvm::cl::desc("Add target-specific passes"),
    llvm::cl::init(true), llvm::cl::cat(qssc::config::getQSSCCategory()));

} // anonymous namespace

llvm::cl::OptionCategory &qssc::config::getQSSCCategory() { return qsscCat_; }

llvm::Error CLIConfigBuilder::populateConfig(QSSConfig &config) {
  if (auto err = populateConfigurationPath_(config))
    return err;

  if (auto err = populateTarget_(config))
    return err;

  if (auto err = populateAllowUnregisteredDialects_(config))
    return err;

  if (auto err = populateAddTargetPasses_(config))
    return err;

  return llvm::Error::success();
}

llvm::Error CLIConfigBuilder::populateConfigurationPath_(QSSConfig &config) {
  if (configurationPath != "")
    config.targetConfigPath = configurationPath;
  return llvm::Error::success();
}

llvm::Error CLIConfigBuilder::populateTarget_(QSSConfig &config) {
  if (targetStr != "")
    config.targetName = targetStr;
  return llvm::Error::success();
}
llvm::Error
CLIConfigBuilder::populateAllowUnregisteredDialects_(QSSConfig &config) {
  config.allowUnregisteredDialects = allowUnregisteredDialects;
  return llvm::Error::success();
}

llvm::Error CLIConfigBuilder::populateAddTargetPasses_(QSSConfig &config) {
  config.addTargetPasses = addTargetPasses;
  return llvm::Error::success();
}
