//===- TargetSystemRegistry.cpp - System Target Registry --------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  Implementation of the QSSC target registry system.
//
//===----------------------------------------------------------------------===//

#include "HAL/TargetSystemRegistry.h"
#include "Parameters/Parameters.h"

using namespace qssc::hal::registry;

namespace {
class NullTarget : public qssc::hal::TargetSystem {
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
} // namespace

TargetSystemInfo *TargetSystemRegistry::nullTargetSystemInfo() {
  static auto nullTarget = std::make_unique<TargetSystemInfo>(
      "NullTarget",
      "A no-op target used by default unless a real target is specified.",
      [](llvm::Optional<llvm::StringRef> config) {
        return std::make_unique<NullTarget>();
      },
      []() { return llvm::Error::success(); },
      []() { return llvm::Error::success(); });
  return nullTarget.get();
}
