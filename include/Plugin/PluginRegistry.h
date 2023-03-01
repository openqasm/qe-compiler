//===- PluginRegistry.h - Plugin Registry -----------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2022, 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  Declaration of the QSSC plugin registry.
//
//===----------------------------------------------------------------------===//
#ifndef PLUGINREGISTRY_H
#define PLUGINREGISTRY_H

//#include "PluginInfo.hpp"
//#include <memory>
//#include <string>
//#include <vector>
//
//#include "llvm/ADT/APFloat.h"
//#include "llvm/ADT/APInt.h"
//#include "llvm/ADT/None.h"
//#include "llvm/ADT/Optional.h"
//#include "llvm/ADT/StringRef.h"
//#include "llvm/Support/Error.h"
//
//#include "HAL/SystemConfiguration.h"
//#include "HAL/TargetSystem.h"
//
//#include "Support/Pimpl.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/ADT/StringMap.h"

namespace qssc::plugin::registry {

    template<typename PluginInfo>
    struct PluginRegistry {

    public:
        template<typename... Args>
        static bool registerPlugin(llvm::StringRef name, Args &&... args) {
            auto &pluginRegistry = instance();
            auto [_, inserted] = pluginRegistry.registry.try_emplace(name, std::forward<Args>(args)...);
            return inserted;
        }

        llvm::Optional<PluginInfo *> lookupPluginInfo(llvm::StringRef pluginName) {
            auto &pluginRegistry = instance();
            auto it = pluginRegistry.registry.find(pluginName);
            if (it == pluginRegistry.registry.end())
                return llvm::None;
            return &it->second;
        }

        // TODO: To be implemented
        // PluginInfo *nullPluginInfo();

        bool pluginExists(llvm::StringRef targetName) {
            auto &pluginRegistry = instance();
            auto it = pluginRegistry.registry.find(targetName);
            return it != pluginRegistry.registry.end();
        }

        const llvm::StringMap<PluginInfo> &registeredPlugins() {
            auto &pluginRegistry = instance();
            return *pluginRegistry.registry;
        }

    private:
        PluginRegistry() = default;

        static PluginRegistry &instance() {
            static PluginRegistry pluginRegistry;
            return pluginRegistry;
        }

    private:
        llvm::StringMap<PluginInfo> registry;
    };

}

#endif