//===- PluginRegistry.h - Plugin Registry -----------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
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

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/ManagedStatic.h"

#include <optional>

namespace qssc::plugin::registry {

template <typename PluginInfo>
struct PluginRegistry {

public:
  PluginRegistry(const PluginRegistry &) = delete;
  void operator=(const PluginRegistry &) = delete;

  template <typename... Args>
  static bool registerPlugin(llvm::StringRef name, Args &&...args) {
    PluginRegistry &pluginRegistry = instance();
    auto [_, inserted] =
        pluginRegistry.registry.try_emplace(name, std::forward<Args>(args)...);
    return inserted;
  }

  static std::optional<PluginInfo *>
  lookupPluginInfo(llvm::StringRef pluginName) {
    PluginRegistry &pluginRegistry = instance();
    auto it = pluginRegistry.registry.find(pluginName);
    if (it == pluginRegistry.registry.end())
      return std::nullopt;
    return &it->second;
  }

  static bool pluginExists(llvm::StringRef targetName) {
    PluginRegistry &pluginRegistry = instance();
    auto it = pluginRegistry.registry.find(targetName);
    return it != pluginRegistry.registry.end();
  }

  static const llvm::StringMap<PluginInfo> &registeredPlugins() {
    PluginRegistry  const&pluginRegistry = instance();
    return pluginRegistry.registry;
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

} // namespace qssc::plugin::registry

#endif
