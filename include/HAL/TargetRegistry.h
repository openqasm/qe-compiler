//===- TargetRegistry.h - System Target Registry ----------------*- C++ -*-===//
//
// (C) Copyright IBM 2021.
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
#ifndef TARGETREGISTRY_H
#define TARGETREGISTRY_H

#include <memory>
#include <string>
#include <vector>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include "HAL/SystemConfiguration.h"
#include "HAL/TargetSystem.h"

#include "Support/Pimpl.h"

namespace mlir {
class MLIRContext;
class ModuleOp;
} // namespace mlir

namespace qssc::hal::registry {

using TargetSystemFactoryFunction =
    std::function<llvm::Expected<std::unique_ptr<qssc::hal::TargetSystem>>(
        llvm::Optional<llvm::StringRef> configurationPath)>;

using TargetRegisterPassesFunction = std::function<llvm::Error()>;

using TargetRegisterPassPipelinesFunction = std::function<llvm::Error()>;

/// Class to group info about a registered target. Such as how to invoke
/// and a description.
class TargetInfo {
public:
  /// Construct this entry
  TargetInfo(llvm::StringRef name, llvm::StringRef description,
             TargetSystemFactoryFunction targetFactory,
             TargetRegisterPassesFunction passRegistrar,
             TargetRegisterPassPipelinesFunction passPipelineRegistrar);
  ~TargetInfo();
  /// Returns the name used to invoke this target from the CLI.
  llvm::StringRef getTargetName() const { return name; }
  /// Returns the description for this target in the CLI.
  llvm::StringRef getTargetDescription() const { return description; }
  /// Create the target system and register it under the given context.
  llvm::Expected<qssc::hal::TargetSystem *>
  createTarget(mlir::MLIRContext *context,
               llvm::Optional<llvm::StringRef> configurationPath);
  /// Get the target system registered for the given context. First checks for
  /// a target registered exactly for the given context. If no such context is
  /// found, checks if a target is registered under nullptr, and returns
  /// that. If no target is found, an error is returned.
  llvm::Expected<qssc::hal::TargetSystem *>
  getTarget(mlir::MLIRContext *context) const;
  /// Register this target's MLIR passes with the QSSC system.
  /// Should only be called once on initialization.
  llvm::Error registerTargetPasses() const { return passRegistrar(); }
  /// Register this target's MLIR passe pipelines with the QSSC system.
  /// Should only be called once on initialization.
  llvm::Error registerTargetPassPipelines() const {
    return passPipelineRegistrar();
  }
  /// Print the help string for this Target.
  void printHelpStr(size_t indent, size_t descIndent) const;

private:
  struct Impl;
  /// Pointer to private implementation.
  qssc::support::Pimpl<Impl> impl;
  /// The name of this Target to invoke from CLI.
  llvm::StringRef name;
  /// Description of this target.
  llvm::StringRef description;
  /// Target context factory function
  TargetSystemFactoryFunction targetFactory;
  /// Pass registry function
  TargetRegisterPassesFunction passRegistrar;
  /// PassPipeline registry function
  TargetRegisterPassPipelinesFunction passPipelineRegistrar;
};

/// Register a specific target allocator with the QSSC system.
/// Following the MLIR pattern this should normally be used with the
/// TargetRegistration template below for ease of use.
void registerTarget(
    llvm::StringRef name, llvm::StringRef description,
    const TargetSystemFactoryFunction &targetFactory,
    const TargetRegisterPassesFunction &passRegistrar,
    const TargetRegisterPassPipelinesFunction &passPipelineRegistrar);

/// Look up the target info for a target. Returns None if not registered.
llvm::Optional<TargetInfo *> lookupTargetInfo(llvm::StringRef targetName);

/// Get the Null target info.
TargetInfo *nullTargetInfo();

/// Verify the target exists
bool targetExists(llvm::StringRef targetName);

/// Available targets
const llvm::StringMap<TargetInfo> &registeredTargets();

/// A Target registration system modelled after MLIR's pass registration system.
/// Here we allow users to register a target through calling the templated
/// struct. This exposes the target through the CLI and also makes it available
/// for extensible and automated target system construction and compilation
/// functionality that that we will build eventually.
///
/// Register a Target with the target registry.
///
/// The argument is optional and may be used when the Target does
/// not have a standard constructor.
template <typename ConcreteTarget>
struct TargetRegistration {
  /// Register a Target with the target registry.
  /// By providing allocation and registration functions.
  TargetRegistration(
      llvm::StringRef name, llvm::StringRef description,
      const TargetSystemFactoryFunction &targetFactory,
      const TargetRegisterPassesFunction &passRegistrar,
      const TargetRegisterPassPipelinesFunction &passPipelineRegistrar) {
    registerTarget(name, description, targetFactory, passRegistrar,
                   passPipelineRegistrar);
  }
  /// Register a target with static methods registerTargetPasses and
  /// registerTargetPassPipelines available to register passes and pass
  /// pipelines respectively.
  TargetRegistration(llvm::StringRef name, llvm::StringRef description,
                     const TargetSystemFactoryFunction &targetFactory)
      : TargetRegistration(name, description, targetFactory,
                           ConcreteTarget::registerTargetPasses,
                           ConcreteTarget::registerTargetPipelines) {}

}; // struct TargetRegistration

} // namespace qssc::hal::registry
#endif // TARGETREGISTRY_H
