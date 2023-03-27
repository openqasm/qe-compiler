//===- QSSConfig.cpp - Config info --------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  This file implements standard handling of QSSConfiguration options.
//
//===----------------------------------------------------------------------===//

#include "Config/QSSConfig.h"

using namespace qssc::config;


llvm::Expected<QSSConfig> QSSConfigBuilder::buildConfig() {
    QSSConfig config;
    if (auto e = populateConfig(config))
        return e;
    return config;
}
