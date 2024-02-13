//===- ThreadedCompilationManager.h - Threaded Scheduler ------*- C++ -*-===//
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
///  This file declares the classes for the top-level threaded compilation
///  interfaces.
///
//===----------------------------------------------------------------------===//
#ifndef THREADEDCOMPILATIONMANAGER_H
#define THREADEDCOMPILATIONMANAGER_H

#include "Config/QSSConfig.h"
#include "HAL/Compile/TargetCompilationManager.h"

#include <mutex>
#include <string>

namespace qssc::hal::compile {

/// @brief A threaded implementation of a TargetCompilationManager
/// based on the threading pools provided by the mlir::MLIRContext.
/// This enables compilation across disjoint subtree of compilation
/// targets in parallel.
/// If threading is disabled within the MLIRContext the implementation
/// will fall back to a sequential unthreaded version.
/// The threaded implementation relies on MLIR's
/// <a
/// href="https://mlir.llvm.org/docs/PassManagement/#operation-pass">multi-threading
/// assumptions</a>. As compilation is based on the shared MLIRContext's
/// threadpool we are able to safely mix parallel nested passes and parallel
/// target compilation subtrees without oversubscribing the compilation host's
/// cores.
class ThreadedCompilationManager : public TargetCompilationManager {
protected:
  /// Threaded depth first walker for a target system using the current
  /// MLIRContext's threadpool.
  llvm::Error walkTargetThreaded(
      Target *target, mlir::TimingScope &timing,
      const TargetCompilationManager::WalkTargetFunction &walkFunc);
  /// Threaded depth first walker for a target system modules using the current
  /// MLIRContext's threadpool.
  llvm::Error walkTargetModulesThreaded(
      Target *target, mlir::ModuleOp targetModuleOp, mlir::TimingScope &timing,
      const TargetCompilationManager::WalkTargetModulesFunction &walkFunc,
      const TargetCompilationManager::WalkTargetModulesFunction
          &postChildrenCallbackFunc);

  virtual void printIR(llvm::Twine msg, mlir::Operation *op,
                       llvm::raw_ostream &out) override;

public:
  using PMBuilder = std::function<llvm::Error(mlir::PassManager &)>;

  ThreadedCompilationManager(qssc::hal::TargetSystem &target,
                             mlir::MLIRContext *context, PMBuilder pmBuilder);
  virtual ~ThreadedCompilationManager() = default;
  virtual const std::string getName() const override;

  virtual llvm::Error compileMLIR(mlir::ModuleOp moduleOp) override;
  virtual llvm::Error compilePayload(mlir::ModuleOp moduleOp,
                                     qssc::payload::Payload &payload,
                                     bool doCompileMLIR = true) override;

  bool isMultithreadingEnabled() {
    return getContext()->isMultithreadingEnabled();
  }
  llvm::ThreadPool &getThreadPool() { return getContext()->getThreadPool(); }

private:
  // Used to store initialized and registered pass managers
  // with the context prior to compilation.
  std::map<Target *, mlir::PassManager> targetPassManagers_;
  // target pass manager map mutex
  std::shared_mutex targetPassManagersMutex_;

  // ensures we register passes with the context
  // in a threadsafe way.
  std::mutex contextMutex_;

  // ensures we print in order when threading
  std::mutex printIRMutex_;

  /// Prepare pass managers in a threaded way
  /// initializing them with the mlir context safely.
  llvm::Error buildTargetPassManagers_(Target &target,
                                       mlir::TimingScope &timing);
  /// Threadsafe initialization of PM to work around
  /// non-threadsafe registration of dependent dialects.
  /// I (Thomas) believe this is related to the conversation here
  /// https://discourse.llvm.org/t/caching-and-parallel-compilation/67907
  /// and will follow up in the thread.
  void registerPassManagerWithContext_(mlir::PassManager &pm);
  /// Thread safely get the passmanager for a target.
  mlir::PassManager &getTargetPassManager_(Target *target);
  /// Thread safely set the passmanager for a target.
  mlir::PassManager &createTargetPassManager_(Target *target);

  /// Compiles the input module for a single target.
  llvm::Error compileMLIRTarget_(Target &target, mlir::ModuleOp targetModuleOp,
                                 mlir::TimingScope &timing);
  /// Compiles the input payload for a single target.
  llvm::Error compilePayloadTarget_(Target &target,
                                    mlir::ModuleOp targetModuleOp,
                                    qssc::payload::Payload &payload,
                                    mlir::TimingScope &timing,
                                    bool doCompileMLIR);

  PMBuilder pmBuilder;

}; // class THREADEDCOMPILATIONMANAGER

} // namespace qssc::hal::compile
#endif // THREADEDCOMPILATIONMANAGER_H
