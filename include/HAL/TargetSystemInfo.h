//===- TargetSystemInfo.h - System Target Registry --------------*- C++ -*-===//
//
// (C) Copyright IBM 20223.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  Declaration of the QSSC target system info.
//
//===----------------------------------------------------------------------===//
#ifndef TARGETSYSTEMINFO_H
#define TARGETSYSTEMINFO_H

#include "HAL/TargetSystem.h"

#include "Support/Pimpl.h"

#include "Plugin/PluginInfo.hpp"

namespace qssc::hal::registry {

    class TargetSystemInfo : public qssc::plugin::registry::PluginInfo<qssc::hal::TargetSystem> {
        using PluginInfo = qssc::plugin::registry::PluginInfo<qssc::hal::TargetSystem>;
        using PassesFunction = std::function<llvm::Error()>;
        using PassPipelinesFunction = std::function<llvm::Error()>;
    public:
        TargetSystemInfo(llvm::StringRef name, llvm::StringRef description,
                         PluginInfo::PluginFactoryFunction targetFactory,
                         PassesFunction passRegistrar,
                         PassPipelinesFunction passPipelineRegistrar);

        ~TargetSystemInfo();

        llvm::Expected<qssc::hal::TargetSystem *>
        createTarget(mlir::MLIRContext *context,
                     llvm::Optional<PluginInfo::PluginConfiguration> configuration);

        llvm::Expected<qssc::hal::TargetSystem *>
        getTarget(mlir::MLIRContext *context) const;

        llvm::Error registerTargetPasses() const;

        llvm::Error registerTargetPassPipelines() const;

    private:
        struct Impl;

        qssc::support::Pimpl<Impl> impl;

        PassesFunction passRegistrar;

        PassPipelinesFunction passPipelineRegistrar;
    };

} // namespace qssc::hal::registry

#endif
