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
#include "HAL/TargetSystem.h"
//
//#include "Support/Pimpl.h"

#include "Plugin/PluginRegistry.h"
#include "Plugin/PluginInfo.hpp"

namespace qssc::hal::registry {

    using TargetSystemInfo = qssc::plugin::registry::PluginInfo<qssc::hal::TargetSystem>;

    class TargetSystemRegistry : public qssc::plugin::registry::PluginRegistry<TargetSystemInfo> {
        using PluginRegistry = qssc::plugin::registry::PluginRegistry<TargetSystemInfo>;
    public:
        template<typename ConcreteTargetSystem>
        static bool registerPlugin(llvm::StringRef name, llvm::StringRef description,
                                   const TargetSystemInfo::PluginFactoryFunction &pluginFactory) {
            bool inserted = PluginRegistry::registerPlugin(name, description, pluginFactory);
            if (inserted) {
                ConcreteTargetSystem::registerTargetPasses();
                ConcreteTargetSystem::registerTargetPipelines();
            }
            return inserted;
        }
    };

}

#endif
