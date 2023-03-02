//===- TargetSystemRegistry.h - System Target Registry ----------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 20223.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  Declaration of the QSSC target registry system.
//
//===----------------------------------------------------------------------===//
#ifndef TARGETSYSTEMREGISTRY_H
#define TARGETSYSTEMREGISTRY_H

#include "HAL/TargetSystem.h"

#include "Plugin/PluginRegistry.h"
#include "TargetSystemInfo.h"

namespace qssc::hal::registry {

    class TargetSystemRegistry : public qssc::plugin::registry::PluginRegistry<TargetSystemInfo> {
        using PluginRegistry = qssc::plugin::registry::PluginRegistry<TargetSystemInfo>;
    public:
        template<typename ConcreteTargetSystem>
        struct InitRegistry {
            template<typename... Args>
            InitRegistry(llvm::StringRef name, Args &&... args) {
                registered = TargetSystemRegistry::registerPlugin<ConcreteTargetSystem>(name, std::forward<Args>(args)...);
            }

            bool registered = false;
        };

        TargetSystemRegistry(const TargetSystemRegistry&) = delete;
        void operator=(const TargetSystemRegistry&) = delete;

        template<typename ConcreteTargetSystem>
        static bool registerPlugin(llvm::StringRef name, llvm::StringRef description,
                                   const TargetSystemInfo::PluginFactoryFunction &pluginFactory) {
            return PluginRegistry::registerPlugin(name, name, description, pluginFactory,
                                                  ConcreteTargetSystem::registerTargetPasses,
                                                  ConcreteTargetSystem::registerTargetPipelines);
        }

        static TargetSystemInfo *nullTargetSystemInfo();
    };

} // namespace qssc::hal::registry

#endif
