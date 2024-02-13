//===- TargetCompilationManager.cpp ----------------------------*- C++ -*--===//
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

#include "HAL/Compile/ThreadedCompilationManager.h"

#include "HAL/Compile/TargetCompilationManager.h"
#include "HAL/TargetSystem.h"
#include "Payload/Payload.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/Timing.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>

using namespace qssc;
using namespace qssc::hal::compile;

ThreadedCompilationManager::ThreadedCompilationManager(
    qssc::hal::TargetSystem &target, mlir::MLIRContext *context,
    ThreadedCompilationManager::PMBuilder pmBuilder)
    : TargetCompilationManager(target, context),
      pmBuilder(std::move(pmBuilder)) {}

const std::string ThreadedCompilationManager::getName() const {
  return "ThreadedCompilationManager";
}

llvm::Error ThreadedCompilationManager::walkTargetModulesThreaded(
    Target *target, mlir::ModuleOp targetModuleOp, mlir::TimingScope &timing,
    const WalkTargetModulesFunction &walkFunc,
    const WalkTargetModulesFunction &postChildrenCallbackFunc) {

  auto parentTiming = timing.nest(target->getName());

  if (auto err = walkFunc(target, targetModuleOp, parentTiming))
    return err;

  // Get child modules in a non-threaded fashion to preserve
  // MLIR parallelization rules
  auto children = target->getChildren();

  // Check if there are children to walk. If not exit early
  if (children.size()) {

    auto childrenTiming = parentTiming.nest("children");

    std::unordered_map<Target *, mlir::ModuleOp> childrenModules;
    for (auto *childTarget : children) {
      auto childModuleOp = childTarget->getModule(targetModuleOp);
      if (auto err = childModuleOp.takeError())
        return err;
      childrenModules[childTarget] = *childModuleOp;
    }

    auto parallelWalkFunc = [&](Target *childTarget) {
      // Recurse on this target's children in a depth first fashion.

      if (auto err = walkTargetModulesThreaded(
              childTarget, childrenModules[childTarget], childrenTiming,
              walkFunc, postChildrenCallbackFunc)) {
        llvm::errs() << err << "\n";
        return mlir::failure();
      }

      return mlir::success();
    };

    // By utilizing the MLIR parallelism methods, we automatically inherit the
    // multiprocessing settings from the context.
    if (mlir::failed(mlir::failableParallelForEach(getContext(), children,
                                                   parallelWalkFunc)))
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "Problems encountered while walking children of target " +
              target->getName());
  }

  if (auto err = postChildrenCallbackFunc(target, targetModuleOp, parentTiming))
    return err;

  return llvm::Error::success();
}

llvm::Error ThreadedCompilationManager::walkTargetThreaded(
    Target *target, mlir::TimingScope &timing,
    const WalkTargetFunction &walkFunc) {

  auto parentTiming = timing.nest(target->getName());
  if (auto err = walkFunc(target, parentTiming))
    return err;

  auto children = target->getChildren();

  // Check if there are children to walk. If not exit early
  if (!children.size())
    return llvm::Error::success();

  auto childrenTiming = parentTiming.nest("children");

  auto parallelWalkFunc = [&](Target *childTarget) {
    // Recurse on this target's children in a depth first fashion.

    if (auto err = walkTargetThreaded(childTarget, childrenTiming, walkFunc)) {
      llvm::errs() << err << "\n";
      return mlir::failure();
    }

    return mlir::success();
  };

  // By utilizing the MLIR parallelism methods, we automatically inherit the
  // multiprocessing settings from the context.
  if (mlir::failed(mlir::failableParallelForEach(getContext(), children,
                                                 parallelWalkFunc)))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Problems encountered while walking children of target " +
            target->getName());

  return llvm::Error::success();
}

llvm::Error ThreadedCompilationManager::buildTargetPassManagers_(
    Target &target, mlir::TimingScope &timing) {

  auto buildPMTiming = timing.nest("build-target-pass-managers");

  // Create dummy timing scope for pass manager building
  // as this is not significant enough to report for each individual target.
  auto targetsTiming = mlir::TimingScope();

  auto threadedBuildTargetPassManager =
      [&](hal::Target *target, mlir::TimingScope &timing) -> llvm::Error {
    auto &pm = createTargetPassManager_(target);

    if (auto err = pmBuilder(pm))
      return err;

    target->enableTiming(timing);
    if (auto err = target->addPasses(pm))
      return err;
    target->disableTiming();

    registerPassManagerWithContext_(pm);

    return llvm::Error::success();
  };

  auto err = walkTargetThreaded(&getTargetSystem(), targetsTiming,
                                threadedBuildTargetPassManager);
  return err;
}

// Mirroring mlir::PassManager::run() we register all of the pass's dependent
// dialects with the context in a thread-safe way to prevent issues with the
// default non-threadsafe modification of the dialect registry performed by the
// pass manager. See:
// https://github.com/llvm/llvm-project/blob/9423e459875b0dcdf24975976838d651a92f1bdb/mlir/lib/Pass/Pass.cpp#L840-L845
void ThreadedCompilationManager::registerPassManagerWithContext_(
    mlir::PassManager &pm) {
  mlir::DialectRegistry dependentDialects;
  pm.getDependentDialects(dependentDialects);
  auto *context = getContext();

  // NOLINTNEXTLINE(clang-diagnostic-ctad-maybe-unsupported)
  std::unique_lock const lock(contextMutex_);
  context->appendDialectRegistry(dependentDialects);
  for (llvm::StringRef const name : dependentDialects.getDialectNames())
    context->getOrLoadDialect(name);
}

mlir::PassManager &
ThreadedCompilationManager::getTargetPassManager_(Target *target) {
  // NOLINTNEXTLINE(clang-diagnostic-ctad-maybe-unsupported)
  std::shared_lock const lock(targetPassManagersMutex_);
  return targetPassManagers_.at(target);
}

mlir::PassManager &
ThreadedCompilationManager::createTargetPassManager_(Target *target) {
  // NOLINTNEXTLINE(clang-diagnostic-ctad-maybe-unsupported)
  std::unique_lock const lock(targetPassManagersMutex_);
  return targetPassManagers_.emplace(target, getContext()).first->second;
}

llvm::Error ThreadedCompilationManager::compileMLIR(mlir::ModuleOp moduleOp) {

  auto compileMLIRTiming = getTimer("compile-mlir");

  auto &target = getTargetSystem();

  /// Build target pass managers prior to compilation
  /// to ensure thread safety
  if (auto err = buildTargetPassManagers_(target, compileMLIRTiming))
    return err;

  auto threadedCompileMLIRTarget =
      [&](hal::Target *target, mlir::ModuleOp targetModuleOp,
          mlir::TimingScope &timing) -> llvm::Error {
    if (auto err = compileMLIRTarget_(*target, targetModuleOp, timing))
      return err;
    return llvm::Error::success();
  };

  // Placeholder function
  auto postChildrenEmitToPayload =
      [&](hal::Target *target, mlir::ModuleOp targetModuleOp,
          mlir::TimingScope &timing) -> llvm::Error {
    return llvm::Error::success();
  };

  auto targetsTiming = compileMLIRTiming.nest("compile-system");

  auto err = walkTargetModulesThreaded(&target, moduleOp, targetsTiming,
                                       threadedCompileMLIRTarget,
                                       postChildrenEmitToPayload);
  return err;
}

llvm::Error ThreadedCompilationManager::compileMLIRTarget_(
    Target &target, mlir::ModuleOp targetModuleOp, mlir::TimingScope &timing) {
  if (getPrintBeforeAllTargetPasses())
    printIR("IR dump before running passes for target " + target.getName(),
            targetModuleOp, llvm::outs());

  auto targetPassesTiming = timing.nest("passes");
  mlir::PassManager &pm = getTargetPassManager_(&target);
  pm.enableTiming(targetPassesTiming);

  if (mlir::failed(pm.run(targetModuleOp))) {
    if (getPrintAfterTargetCompileFailure())
      printIR("IR dump after failure emitting payload for target " +
                  target.getName(),
              targetModuleOp, llvm::outs());
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Problems running the pass pipeline for target " + target.getName());
  }

  if (getPrintAfterAllTargetPasses())
    printIR("IR dump after running passes for target " + target.getName(),
            targetModuleOp, llvm::outs());

  return llvm::Error::success();
}

llvm::Error
ThreadedCompilationManager::compilePayload(mlir::ModuleOp moduleOp,
                                           qssc::payload::Payload &payload,
                                           bool doCompileMLIR) {

  auto compilePayloadTiming = getTimer("compile-payload");

  auto &target = getTargetSystem();

  /// Build target pass managers prior to compilation
  /// to ensure thread safety
  if (auto err = buildTargetPassManagers_(target, compilePayloadTiming))
    return err;

  auto threadedCompilePayloadTarget =
      [&](hal::Target *target, mlir::ModuleOp targetModuleOp,
          mlir::TimingScope &timing) -> llvm::Error {
    if (auto err = compilePayloadTarget_(*target, targetModuleOp, payload,
                                         timing, doCompileMLIR))
      return err;
    return llvm::Error::success();
  };

  auto postChildrenEmitToPayload =
      [&](hal::Target *target, mlir::ModuleOp targetModuleOp,
          mlir::TimingScope &timing) -> llvm::Error {
    auto emitToPayloadTiming = timing.nest("emit-to-payload-post-children");
    target->enableTiming(emitToPayloadTiming);
    if (auto err = target->emitToPayloadPostChildren(targetModuleOp, payload))
      return err;
    target->disableTiming();

    return llvm::Error::success();
  };

  auto targetsTiming = compilePayloadTiming.nest("compile-system");
  auto err = walkTargetModulesThreaded(&target, moduleOp, targetsTiming,
                                       threadedCompilePayloadTarget,
                                       postChildrenEmitToPayload);
  return err;
}

llvm::Error ThreadedCompilationManager::compilePayloadTarget_(
    Target &target, mlir::ModuleOp targetModuleOp,
    qssc::payload::Payload &payload, mlir::TimingScope &timing,
    bool doCompileMLIR) {

  if (doCompileMLIR)
    if (auto err = compileMLIRTarget_(target, targetModuleOp, timing))
      return err;

  if (getPrintBeforeAllTargetPayload())
    printIR("IR dump before emitting payload for target " + target.getName(),
            targetModuleOp, llvm::outs());

  auto emitToPayloadTiming = timing.nest("emit-to-payload");
  target.enableTiming(emitToPayloadTiming);
  if (auto err = target.emitToPayload(targetModuleOp, payload)) {
    if (getPrintAfterTargetCompileFailure())
      printIR("IR dump after failure emitting payload for target " +
                  target.getName(),
              targetModuleOp, llvm::outs());
    return err;
  }
  target.disableTiming();

  return llvm::Error::success();
}

void ThreadedCompilationManager::printIR(llvm::Twine msg, mlir::Operation *op,
                                         llvm::raw_ostream &out) {
  const std::lock_guard<std::mutex> lock(printIRMutex_);
  TargetCompilationManager::printIR(msg, op, out);
}
