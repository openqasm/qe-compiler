//===- TargetOperationPass.h - Common target pass class  --------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// Common operations for target passes.
///
//===----------------------------------------------------------------------===//

#ifndef TARGET_OPERATION_PASS_H
#define TARGET_OPERATION_PASS_H

#include "HAL/TargetSystemRegistry.h"
#include "QSSC.h"

#include "Dialect/QUIR/IR/QUIROps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace qssc::hal {

/// Baseclass inheriting from OperationPass containing common code-generation
/// helpers for QSSC targets.
template <typename TargetT, typename OpT = void>
class TargetOperationPass : public mlir::OperationPass<OpT> {

protected:
  explicit TargetOperationPass(mlir::TypeID passID)
      : mlir::OperationPass<OpT>(passID) {}

  void runOnOperation() override final {
    if (auto *target = getTargetSystemOrFail())
      runOnOperation(*target);
  }

  virtual void runOnOperation(TargetT &target) = 0;

  /**
   * Gets the target system.
   * @return A non-owning pointer to the target system.
   */
  TargetT *getTargetSystemOrFail() {
    auto targetInfo =
        registry::TargetSystemRegistry::lookupPluginInfo(TargetT::name);
    if (!targetInfo) {
      llvm::errs() << "Error: target '" << TargetT::name
                   << "' is not registered.\n";
      mlir::OperationPass<OpT>::signalPassFailure();
      return nullptr;
    }

    auto target =
        targetInfo.value()->getTarget(&mlir::OperationPass<OpT>::getContext());
    if (!target) {
      // look for a child target that matches
      for (const auto &childName : TargetT::childNames) {
        targetInfo = registry::TargetSystemRegistry::lookupPluginInfo(childName);
        target = targetInfo.value()->getTarget(
                 &mlir::OperationPass<OpT>::getContext());
        if (targetInfo && target)
          break;
      }
      if (!target) {
        llvm::errs() << "Error: failed to get target '" << TargetT::name
                     << "':\n";
        llvm::errs() << target.takeError();
        mlir::OperationPass<OpT>::signalPassFailure();
        return nullptr;
      }
    }

    auto *castedTarget = dynamic_cast<TargetT *>(target.get());
    if (!castedTarget) {
      llvm::errs() << "Error: target registered as '" << TargetT::name
                   << "' does not have the expected type.\n";
      mlir::OperationPass<OpT>::signalPassFailure();
      return nullptr;
    }

    return castedTarget;
  }
};

} // namespace qssc::hal

#endif // TARGET_OPERATION_PASS_H
