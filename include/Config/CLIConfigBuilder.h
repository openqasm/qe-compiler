//===- CLIConfig.h - CLI Configuration builder ------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  Populate the configuration from the CLI.
//
//===----------------------------------------------------------------------===//
#ifndef QSSC_CLICONFIG_H
#define QSSC_CLICONFIG_H

#include "Config/QSSConfig.h"

namespace qssc::config {

class CLIConfigBuilder : QSSConfigBuilder {
    llvm::Error populateConfig(QSSConfig &config) override;
};


} // namespace qssc::config
#endif // QSS_CLICONFIG_H
