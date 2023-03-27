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


/// @brief The QSS configuration data structure that is to be used for global
/// configuration of the QSS infrastructure. This is to be used for static
/// options that are rarely changed for a system and do not need to be dynamically
/// extensible (such as pluggable TargetInstrument and their configuration).
/// This configuration is constructed from several sources such as CLI,
/// environment variables and possible configuration file formats through
/// QSSConfigBuilder implementations which apply successive views over the
/// configuration to produce the final configuration.
struct QSSConfig {
    /// @brief The TargetSystem to target compilation for.
    std::optional<std::string> targetName = std::nullopt;
    /// @brief The path to the TargetSystem configuration information.
    std::optional<std::string> targetConfigPath = std::nullopt;
    /// @brief Allow unregistered dialects to be used during compilation.
    bool allowUnregisteredDialects = false;
    /// @brief Register target passes with the compiler.
    bool addTargetPasses = true;
};

/// @brief A builder class for the QSSConfig. All standard configuration
/// population should be completed through builders.
class QSSConfigBuilder {
    public:
        /// Build a new QSSConfig just from this builder
        virtual llvm::Expected<QSSConfig> buildConfig();
        /// Populate an existing QSSConfig from this builder.
        /// This may layer on top of existing configuration settings.
        virtual llvm::Error populateConfig(QSSConfig &config) = 0;
        virtual ~QSSConfigBuilder() = default;
};


} // namespace qssc::config
#endif // QSS_QSSCONFIG_H
