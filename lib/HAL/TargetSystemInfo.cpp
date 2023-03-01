//===- TargetSystemInfo.cpp - System Target Info ----------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  Implementation of the QSSC target system info.
//
//===----------------------------------------------------------------------===//

#include "HAL/TargetSystemInfo.h"

#include <memory>

namespace qssc::hal::registry {

    struct TargetSystemInfo::Impl {
        llvm::DenseMap<mlir::MLIRContext *, std::unique_ptr<TargetSystem>> managedTargets{};
    };

    TargetSystemInfo::TargetSystemInfo(llvm::StringRef name, llvm::StringRef description,
                                       PluginInfo::PluginFactoryFunction targetFactory,
                                       PassesFunction passRegistrar,
                                       PassPipelinesFunction passPipelineRegistrar)
            : TargetSystemInfo::PluginInfo(name, description, std::move(targetFactory)),
              impl(std::make_unique<Impl>()),
              passRegistrar(std::move(passRegistrar)),
              passPipelineRegistrar(std::move(passPipelineRegistrar)) {}

    TargetSystemInfo::~TargetSystemInfo() = default;

    llvm::Expected<qssc::hal::TargetSystem *>
    TargetSystemInfo::createTarget(mlir::MLIRContext *context,
                                   llvm::Optional<PluginInfo::PluginConfiguration> configuration) {
        auto target = PluginInfo::createPluginInstance(configuration);
        if (!target) {
            return target.takeError();
        }
        impl->managedTargets[context] = std::move(target.get());
        return impl->managedTargets[context].get();
    }

    llvm::Expected<qssc::hal::TargetSystem *>
    TargetSystemInfo::getTarget(mlir::MLIRContext *context) const {
        auto it = impl->managedTargets.find(context);
        if (it != impl->managedTargets.end())
            return it->getSecond().get();

        // Check if a default value exists.
        it = impl->managedTargets.find(nullptr);
        if (it != impl->managedTargets.end())
            return it->getSecond().get();

        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "Error: no target of type '" + getName() +
                                       "' registered for the given context.\n");
    }

    llvm::Error TargetSystemInfo::registerTargetPasses() const {
        return passRegistrar();
    }

    llvm::Error TargetSystemInfo::registerTargetPassPipelines() const {
        return passPipelineRegistrar();
    }
} // namespace qssc::hal::registry
