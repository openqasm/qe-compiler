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

#include "API/errors.h"
#include "HAL/TargetSystem.h"
#include "HAL/TargetSystemInfo.h"
#include "Payload/Payload.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <utility>

using namespace qssc::hal::registry;

namespace {
class NullTarget : public qssc::hal::TargetSystem {
public:
  NullTarget() : TargetSystem("NullTarget", nullptr, std::nullopt) {}

  // Do nothing.
  llvm::Error addPasses(mlir::PassManager &pm) override {
    return llvm::Error::success();
  }

  // Do nothing.
  llvm::Error emitToPayload(mlir::ModuleOp moduleOp,
                            qssc::payload::Payload &payload) override {
    return llvm::Error::success();
  }
};
} // namespace

TargetSystemInfo *TargetSystemRegistry::nullTargetSystemInfo() {
  static auto nullTarget = std::make_unique<TargetSystemInfo>(
      "NullTarget",
      "A no-op target used by default unless a real target is specified.",
      [](std::optional<std::pair<llvm::StringRef, qssc::OptDiagnosticCallback>>
             const &config) { return std::make_unique<NullTarget>(); },
      []() { return llvm::Error::success(); },
      []() { return llvm::Error::success(); });
  return nullTarget.get();
}
