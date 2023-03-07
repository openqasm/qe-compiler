//===- TargetRegistry.cpp - System Target Registry --------------*- C++ -*-===//
//
// (C) Copyright IBM 2021 - 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
//  Implementation of the QSSC target registry system.
//
//===----------------------------------------------------------------------===//

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/ManagedStatic.h"

#include "mlir/IR/MLIRContext.h"

#include "HAL/TargetRegistry.h"

#include <utility>

// Inject static initialization headers from targets.
#include "Targets.inc"

using namespace qssc::hal;
using namespace qssc::hal::registry;

/// Static mapping of all of the registered targets.
static llvm::ManagedStatic<llvm::StringMap<TargetInfo>> targetRegistry;

/// This is the implementation class (following the Pimpl idiom) for
/// TargetInfo, which encapsulates all of its implementation-specific
/// members.
/// Details: https://en.cppreference.com/w/cpp/language/pimpl
struct TargetInfo::Impl {
  llvm::DenseMap<mlir::MLIRContext *, std::unique_ptr<TargetSystem>>
      managedTargets{};
};

TargetInfo::TargetInfo(
    llvm::StringRef name, llvm::StringRef description,
    TargetSystemFactoryFunction targetFactory,
    TargetRegisterPassesFunction passRegistrar,
    TargetRegisterPassPipelinesFunction passPipelineRegistrar)
    : impl(std::make_unique<Impl>()), name(name), description(description),
      targetFactory(std::move(targetFactory)),
      passRegistrar(std::move(passRegistrar)),
      passPipelineRegistrar(std::move(passPipelineRegistrar)) {}

TargetInfo::~TargetInfo() = default;

llvm::Expected<TargetSystem *>
TargetInfo::createTarget(mlir::MLIRContext *context,
                         llvm::Optional<llvm::StringRef> configurationPath) {
  auto target = targetFactory(configurationPath);
  if (!target)
    return target.takeError();

  impl->managedTargets[context] = std::move(target.get());
  return impl->managedTargets[context].get();
}

llvm::Expected<TargetSystem *>
TargetInfo::getTarget(mlir::MLIRContext *context) const {
  auto it = impl->managedTargets.find(context);
  if (it != impl->managedTargets.end())
    return it->getSecond().get();

  // Check if a default value exists.
  it = impl->managedTargets.find(nullptr);
  if (it != impl->managedTargets.end())
    return it->getSecond().get();

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "Error: no target of type '" + this->name +
                                     "' registered for the given context.\n");
}

/// Print the help information for this target. This includes the name,
/// description. `descIndent` is the indent that the
/// descriptions should be aligned.
void TargetInfo::printHelpStr(size_t indent, size_t descIndent) const {
  size_t numSpaces = descIndent - indent - 4;
  llvm::outs().indent(indent)
      << "--" << llvm::left_justify(getTargetName(), numSpaces) << "- "
      << getTargetDescription() << '\n';
}

void registry::registerTarget(
    llvm::StringRef name, llvm::StringRef description,
    const TargetSystemFactoryFunction &targetFactory,
    const TargetRegisterPassesFunction &passRegistrar,
    const TargetRegisterPassPipelinesFunction &passPipelineRegistrar) {
  targetRegistry->try_emplace(name, name, description, targetFactory,
                              passRegistrar, passPipelineRegistrar);
}

/// Returns the target info for the specified target name.
llvm::Optional<TargetInfo *>
registry::lookupTargetInfo(llvm::StringRef targetName) {
  auto it = targetRegistry->find(targetName);
  if (it == targetRegistry->end())
    return llvm::None;
  return &it->second;
}

class NullTarget : public TargetSystem {
public:
  NullTarget() : TargetSystem("NullTarget", nullptr) {}

  // Do nothing.
  llvm::Error addPayloadPasses(mlir::PassManager &pm) override {
    return llvm::Error::success();
  }

  // Do nothing.
  llvm::Error addToPayload(mlir::ModuleOp &moduleOp,
                           qssc::payload::Payload &payload) override {
    return llvm::Error::success();
  }
};

/// Get the Null target info.
TargetInfo *registry::nullTargetInfo() {
  static auto nullTarget = std::make_unique<TargetInfo>(
      "NullTarget",
      "A no-op target used by default unless a real target is specified.",
      [](llvm::Optional<llvm::StringRef> config) {
        return std::make_unique<NullTarget>();
      },
      []() { return llvm::Error::success(); },
      []() { return llvm::Error::success(); });
  return nullTarget.get();
}

/// Returns the target info for the specified target name.
bool registry::targetExists(llvm::StringRef targetName) {
  auto it = targetRegistry->find(targetName);
  return it != targetRegistry->end();
}

const llvm::StringMap<TargetInfo> &registry::registeredTargets() {
  return *targetRegistry;
}
