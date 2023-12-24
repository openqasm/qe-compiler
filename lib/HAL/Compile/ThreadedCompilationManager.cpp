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

#include "mlir/IR/Threading.h"

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
    Target *target, mlir::ModuleOp targetModuleOp,
    const WalkTargetModulesFunction &walkFunc,
    const WalkTargetModulesFunction &postChildrenCallbackFunc) {

  if (auto err = walkFunc(target, targetModuleOp))
    return err;

  // Get child modules in a non-threaded fashion to preserve
  // MLIR parallelization rules
  auto children = target->getChildren();
  std::unordered_map<Target *, mlir::ModuleOp> childrenModules;
  for (auto *childTarget : children) {
    auto childModuleOp = childTarget->getModule(targetModuleOp);
    if (auto err = childModuleOp.takeError())
      return err;
    childrenModules[childTarget] = *childModuleOp;
  }

  auto parallelWalkFunc = [&](Target *childTarget) {
    // Recurse on this target's children in a depth first fashion.

    if (auto err = walkTargetModulesThreaded(childTarget, childrenModules[childTarget],
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

  if (auto err = postChildrenCallbackFunc(target, targetModuleOp))
    return err;

  return llvm::Error::success();
}

llvm::Error ThreadedCompilationManager::walkTargetThreaded(
    Target *target, const WalkTargetFunction &walkFunc) {

  if (auto err = walkFunc(target))
    return err;

  auto parallelWalkFunc = [&](Target *childTarget) {
    // Recurse on this target's children in a depth first fashion.

    if (auto err = walkTargetThreaded(childTarget, walkFunc)) {
      llvm::errs() << err << "\n";
      return mlir::failure();
    }

    return mlir::success();
  };

  // By utilizing the MLIR parallelism methods, we automatically inherit the
  // multiprocessing settings from the context.
  if (mlir::failed(mlir::failableParallelForEach(getContext(), target->getChildren(),
                                                 parallelWalkFunc)))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Problems encountered while walking children of target " +
            target->getName());

  return llvm::Error::success();
}

llvm::Error ThreadedCompilationManager::buildTargetPassManagers_(Target &target) {

  auto threadedBuildTargetPassManager =
      [&](hal::Target *target) -> llvm::Error {

    auto &pm = createTargetPassManager_(target);

    if (auto err = pmBuilder(pm))
      return err;

    if (auto err = target->addPasses(pm))
      return err;

    registerPassManagerWithContext_(pm);

    return llvm::Error::success();
  };

  return walkTargetThreaded(&getTargetSystem(), threadedBuildTargetPassManager);
}

// Mirroring mlir::PassManager::run() we register all of the pass's dependent dialects
// with the context in a thread-safe way to prevent issues with the default non-threadsafe
// modification of the dialect registry performed by the pass manager.
// See: https://github.com/llvm/llvm-project/blob/9423e459875b0dcdf24975976838d651a92f1bdb/mlir/lib/Pass/Pass.cpp#L840-L845
void ThreadedCompilationManager::registerPassManagerWithContext_(mlir::PassManager &pm) {
  mlir::DialectRegistry dependentDialects;
  pm.getDependentDialects(dependentDialects);
  auto *context = getContext();

  std::unique_lock lock(contextMutex_);
  context->appendDialectRegistry(dependentDialects);
  for (llvm::StringRef name : dependentDialects.getDialectNames())
    context->getOrLoadDialect(name);
}

mlir::PassManager& ThreadedCompilationManager::getTargetPassManager_(Target *target) {
  std::shared_lock lock(targetPassManagersMutex_);
  return targetPassManagers_.at(target);
}

mlir::PassManager& ThreadedCompilationManager::createTargetPassManager_(Target *target) {
  std::unique_lock lock(targetPassManagersMutex_);
  return targetPassManagers_.emplace(target, getContext()).first->second;
}

llvm::Error ThreadedCompilationManager::compileMLIR(mlir::ModuleOp moduleOp) {

  auto &target = getTargetSystem();

  /// Build target pass managers prior to compilation
  /// to ensure thread safety
  if (auto err = buildTargetPassManagers_(target))
    return err;

  auto threadedCompileMLIRTarget =
      [&](hal::Target *target, mlir::ModuleOp targetModuleOp) -> llvm::Error {
    if (auto err = compileMLIRTarget_(*target, targetModuleOp))
      return err;
    return llvm::Error::success();
  };

  // Placeholder function
  auto postChildrenEmitToPayload =
      [&](hal::Target *target, mlir::ModuleOp targetModuleOp) -> llvm::Error {
    return llvm::Error::success();
  };

  return walkTargetModulesThreaded(&target, moduleOp,
                            threadedCompileMLIRTarget,
                            postChildrenEmitToPayload);
}

llvm::Error
ThreadedCompilationManager::compileMLIRTarget_(Target &target,
                                              mlir::ModuleOp targetModuleOp) {
  if (getPrintBeforeAllTargetPasses())
    printIR("IR dump before running passes for target " + target.getName(),
            targetModuleOp, llvm::outs());

  mlir::PassManager &pm = getTargetPassManager_(&target);

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
  auto &target = getTargetSystem();

  /// Build target pass managers prior to compilation
  /// to ensure thread safety
  if (auto err = buildTargetPassManagers_(target))
    return err;


  auto threadedCompilePayloadTarget =
      [&](hal::Target *target, mlir::ModuleOp targetModuleOp) -> llvm::Error {
    if (auto err = compilePayloadTarget_(*target, targetModuleOp, payload,
                                        doCompileMLIR))
      return err;
    return llvm::Error::success();
  };

  auto postChildrenEmitToPayload =
      [&](hal::Target *target, mlir::ModuleOp targetModuleOp) -> llvm::Error {
    if (auto err = target->emitToPayloadPostChildren(targetModuleOp, payload))
      return err;
    return llvm::Error::success();
  };

  return walkTargetModulesThreaded(&target, moduleOp,
                            threadedCompilePayloadTarget,
                            postChildrenEmitToPayload);
}

llvm::Error ThreadedCompilationManager::compilePayloadTarget_(
    Target &target, mlir::ModuleOp targetModuleOp,
    qssc::payload::Payload &payload, bool doCompileMLIR) {

  if (doCompileMLIR)
    if (auto err = compileMLIRTarget_(target, targetModuleOp))
      return err;

  if (getPrintBeforeAllTargetPayload())
    printIR("IR dump before emitting payload for target " + target.getName(),
            targetModuleOp, llvm::outs());

  if (auto err = target.emitToPayload(targetModuleOp, payload)) {
    if (getPrintAfterTargetCompileFailure())
      printIR("IR dump after failure emitting payload for target " +
                  target.getName(),
              targetModuleOp, llvm::outs());
    return err;
  }
  return llvm::Error::success();
}

void ThreadedCompilationManager::printIR(llvm::StringRef msg, mlir::Operation *op,
                                       llvm::raw_ostream &out) {
  const std::lock_guard<std::mutex> lock(printIRMutex_);
  TargetCompilationManager::printIR(msg, op, out);
}
