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
//  Declaration of the QSSC plugin info.
//
//===----------------------------------------------------------------------===//
#ifndef PLUGININFO_H
#define PLUGININFO_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace qssc::plugin::registry {

template <typename TPluginType>
class PluginInfo {
public:
  using PluginType = TPluginType;
  using PluginConfiguration = typename PluginType::PluginConfiguration;
  using PluginFactoryFunction =
      std::function<llvm::Expected<std::unique_ptr<PluginType>>(
          std::optional<PluginConfiguration> configuration)>;

public:
  PluginInfo(llvm::StringRef name, llvm::StringRef description,
             PluginFactoryFunction factoryFunction)
      : name(name), description(description),
        factoryFunction(std::move(factoryFunction)) {}

  ~PluginInfo() = default;

  [[nodiscard]] llvm::StringRef getName() const { return name; }

  [[nodiscard]] llvm::StringRef getDescription() const { return description; }

  /// Returns a new instance of the registered PluginType
  llvm::Expected<std::unique_ptr<PluginType>>
  createPluginInstance(std::optional<PluginConfiguration> configuration) {
    return factoryFunction(configuration);
  }

private:
  llvm::StringRef name;
  llvm::StringRef description;
  PluginFactoryFunction factoryFunction;
};

/// Print the help string for the given PluginInfo<TPluginType>.
template <typename TPluginType>
void printHelpStr(const PluginInfo<TPluginType> &pluginInfo, size_t indent,
                  size_t descIndent) {
  const size_t numSpaces = descIndent - indent - 4;
  llvm::outs().indent(indent)
      << "--" << llvm::left_justify(pluginInfo.getName(), numSpaces) << "- "
      << pluginInfo.getDescription() << '\n';
}

} // namespace qssc::plugin::registry

#endif
