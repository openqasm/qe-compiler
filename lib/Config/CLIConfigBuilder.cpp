//===- CLIConfigBuilder.cpp - QSSConfig from the CLI ------*- C++ -*--------===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  This file implements building the configuration from the CLI.
//
//===----------------------------------------------------------------------===//

#include "Config/CLIConfigBuilder.h"

using namespace qssc::config;

llvm::Error CLIConfigBuilder::populateConfig(QSSConfig &config) {
    return llvm::Error::success();
}
