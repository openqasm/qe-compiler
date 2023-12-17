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

llvm::Error ThreadedCompilationManager::walkTargetThreaded(
    Target *target, mlir::ModuleOp targetModuleOp,
    const WalkTargetFunction &walkFunc, const WalkTargetFunction &postChildrenCallbackFunc) {

  if (auto err = walkFunc(target, targetModuleOp))
    return err;

  // Get child modules in a non-threaded fashion to preserve
  // MLIR parallelization rules

  auto parallelWalkFunc = [&](Target *childTarget) {
    // Recurse on this target's children in a depth first fashion.
    auto childModuleOp = childTarget->getModule(targetModuleOp);
    if (auto err = childModuleOp.takeError()) {
      llvm::errs() << err << "\n";
      return mlir::failure();
    }

    if (auto err = walkTargetThreaded(childTarget, *childModuleOp, walkFunc, postChildrenCallbackFunc)) {
      llvm::errs() << err << "\n";
      return mlir::failure();
    }

    return mlir::success();
  };

  // By utilizing the MLIR parallelism methods, we automatically inherit the
  // multiprocessing settings from the context.
  if (mlir::failed(mlir::failableParallelForEach(
          getContext(), target->getChildren(), parallelWalkFunc)))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Problems encountered while walking children of target " +
            target->getName());

  if (auto err = postChildrenCallbackFunc(target, targetModuleOp))
    return err;

  return llvm::Error::success();
}

llvm::Error
ThreadedCompilationManager::buildTargetPassManager(mlir::PassManager &pm) {
  return pmBuilder(pm);
}

llvm::Error ThreadedCompilationManager::compileMLIR(mlir::ModuleOp moduleOp) {

  auto threadedCompileMLIRTarget =
      [&](hal::Target *target, mlir::ModuleOp targetModuleOp) -> llvm::Error {
    if (auto err = compileMLIRTarget(*target, targetModuleOp))
      return err;
    return llvm::Error::success();
  };

  // Placeholder function
  auto postChildrenEmitToPayload =
      [&](hal::Target *target, mlir::ModuleOp targetModuleOp) -> llvm::Error {
    return llvm::Error::success();
  };


  return walkTargetThreaded(&getTargetSystem(), moduleOp,
                            threadedCompileMLIRTarget,
                            postChildrenEmitToPayload);
}

llvm::Error
ThreadedCompilationManager::compileMLIRTarget(Target &target,
                                              mlir::ModuleOp targetModuleOp) {
  mlir::PassManager pm(getContext());
  if (auto err = buildTargetPassManager(pm))
    return err;

  if (auto err = target.addPasses(pm))
    return err;

  if (getPrintBeforeAllTargetPasses())
    printIR("IR dump before running passes for target " + target.getName(),
            targetModuleOp, llvm::outs());

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
  auto threadedCompilePayloadTarget =
      [&](hal::Target *target, mlir::ModuleOp targetModuleOp) -> llvm::Error {
    if (auto err = compilePayloadTarget(*target, targetModuleOp, payload,
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


  return walkTargetThreaded(&getTargetSystem(), moduleOp,
                            threadedCompilePayloadTarget,
                            postChildrenEmitToPayload);
}

llvm::Error ThreadedCompilationManager::compilePayloadTarget(
    Target &target, mlir::ModuleOp targetModuleOp,
    qssc::payload::Payload &payload, bool doCompileMLIR) {

  if (doCompileMLIR)
    if (auto err = compileMLIRTarget(target, targetModuleOp))
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
