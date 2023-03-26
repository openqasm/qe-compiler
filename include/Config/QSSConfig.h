//===- QSSConfig.h - Global QSS config ----------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  A centralized API for configuration handling within the QSS infrastructure.
//
//===----------------------------------------------------------------------===//
#ifndef QSSC_QSSCONFIG_H
#define QSSC_QSSCONFIG_H

#include "llvm/Support/Error.h"

#include <optional>
#include <string>

namespace qssc::config {

struct QSSConfig {
    std::optional<std::string> targetName = std::nullopt;
    std::optional<std::string> targetConfigPath = std::nullopt;
};

/// @brief A builder class for the QSSConfig. All standard configuration
/// population should be completed through builders.
class QSSConfigBuilder {
    public:
        /// Build a new QSSConfig just from this builder
        virtual llvm::Expected<QSSConfig> buildConfig();
        /// Populate an existing QSSConfig from this builder.
        /// This may layer on top of existing configuration settings.
        virtual llvm::Error populateConfig(QSSConfig &config);
        virtual ~QSSConfigBuilder() = default;
};


} // namespace qssc::config
#endif // QSS_QSSCONFIG_H
