//===- TargetSystemRegistry.h - System Target Registry ----------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
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

class TargetSystemRegistry
    : public qssc::plugin::registry::PluginRegistry<TargetSystemInfo> {
  using PluginRegistry =
      qssc::plugin::registry::PluginRegistry<TargetSystemInfo>;

public:
  TargetSystemRegistry(const TargetSystemRegistry &) = delete;
  void operator=(const TargetSystemRegistry &) = delete;

  /// Register a specific target allocator with the QSSC system.
  template <typename ConcreteTargetSystem>
  static bool
  registerPlugin(llvm::StringRef name, llvm::StringRef description,
                 const TargetSystemInfo::PluginFactoryFunction &pluginFactory) {
    return PluginRegistry::registerPlugin(
        name, name, description, pluginFactory,
        ConcreteTargetSystem::registerTargetPasses,
        ConcreteTargetSystem::registerTargetPipelines);
  }

  static TargetSystemInfo *nullTargetSystemInfo();
};

} // namespace qssc::hal::registry

#endif
