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

    template<typename TPluginType>
    class PluginInfo {
    public:
        using PluginType = TPluginType;
        using PluginConfiguration = typename PluginType::PluginConfiguration;
        using PluginFactoryFunction = std::function<llvm::Expected<std::unique_ptr<PluginType>>(
                llvm::Optional<PluginConfiguration> configuration)>;

    public:
        PluginInfo(llvm::StringRef name, llvm::StringRef description,
                   PluginFactoryFunction factoryFunction) : name(name), description(description),
                                                            factoryFunction(std::move(factoryFunction)) {}

        ~PluginInfo() = default;

        [[nodiscard]] llvm::StringRef getName() const { return name; }

        [[nodiscard]] llvm::StringRef getDescription() const { return description; }

        llvm::Expected<std::unique_ptr<PluginType>>
        createPluginInstance(llvm::Optional<PluginConfiguration> configuration = llvm::None) {
            return factoryFunction(configuration);
        }

    private:
        llvm::StringRef name;
        llvm::StringRef description;
        PluginFactoryFunction factoryFunction;
    };

    // TODO: Maybe move somewhere else, closer to CLI. This doesn't belong to the plugin registry framework
    template<typename TPluginType>
    void printHelpStr(const PluginInfo<TPluginType> &pluginInfo, size_t indent, size_t descIndent) {
        size_t numSpaces = descIndent - indent - 4;
        llvm::outs().indent(indent)
                << "--" << llvm::left_justify(pluginInfo.getName(), numSpaces) << "- "
                << pluginInfo.getDescription() << '\n';
    }

} // namespace qssc::plugin::registry

#endif