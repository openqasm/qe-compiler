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

#include "HAL/TargetSystem.h"

#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <utility>

// Inject static initialization headers from targets. We need to include them in
// a translation unit that is not being optimized (removed) by the compiler.
// NOLINTNEXTLINE: Required for target initializations even if not used
#include "Targets.inc"

using namespace qssc::hal::registry;

/// This is the implementation class (following the Pimpl idiom) for
/// TargetSystemInfo, which encapsulates all of its implementation-specific
/// members.
/// Details: https://en.cppreference.com/w/cpp/language/pimpl
struct TargetSystemInfo::Impl {
  llvm::DenseMap<mlir::MLIRContext *, std::unique_ptr<TargetSystem>>
      managedTargets{};
};

TargetSystemInfo::TargetSystemInfo(
    llvm::StringRef name, llvm::StringRef description,
    PluginInfo::PluginFactoryFunction targetFactory,
    PassesFunction passRegistrar, PassPipelinesFunction passPipelineRegistrar)
    : TargetSystemInfo::PluginInfo(name, description, std::move(targetFactory)),
      impl(std::make_unique<Impl>()), passRegistrar(std::move(passRegistrar)),
      passPipelineRegistrar(std::move(passPipelineRegistrar)) {}

TargetSystemInfo::~TargetSystemInfo() = default;

llvm::Expected<qssc::hal::TargetSystem *> TargetSystemInfo::createTarget(
    mlir::MLIRContext *context,
    std::optional<PluginInfo::PluginConfiguration> configuration) {
  auto target = PluginInfo::createPluginInstance(std::move(configuration));
  if (!target)
    return target.takeError();
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
