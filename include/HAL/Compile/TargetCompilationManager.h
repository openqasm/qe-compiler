//===- TargetCompilationManager.h - Compilation Scheduler -----*- C++ -*-===//
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
///  This file declares the classes for the top-level compilation scheduling
///  interface.
///
//===----------------------------------------------------------------------===//
#ifndef TARGETCOMPILATIONMANAGER_H
#define TARGETCOMPILATIONMANAGER_H

#include "HAL/TargetSystem.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Error.h"

#include <string>

using namespace qssc;

namespace qssc::hal::compile {

/// @brief Base class for the compiler's
/// target compilation infrastructure.
/// A target system is a tree of compilation targets.
/// We aim to support compiling each disjoint
/// target subtree independently.
class TargetCompilationManager {
protected:
  TargetCompilationManager(hal::TargetSystem &target,
                           mlir::MLIRContext *context);

  using WalkTargetModulesFunction =
      std::function<llvm::Error(hal::Target *, mlir::ModuleOp)>;
  using WalkTargetFunction =
      std::function<llvm::Error(hal::Target *)>;

  // Depth first walker for a target system
  llvm::Error walkTarget(Target *target, const WalkTargetFunction &walkFunc);
  // Depth first walker for a target system modules
  llvm::Error walkTargetModules(Target *target, mlir::ModuleOp targetModuleOp,
                         const WalkTargetModulesFunction &walkFunc,
                         const WalkTargetModulesFunction &postChildrenCallbackFunc);

public:
  virtual ~TargetCompilationManager() = default;
  virtual const std::string getName() const = 0;

  /// Get the base target system to be compiled.
  virtual hal::Target &getTargetSystem() { return target; }

  /// Get the base MLIR context for this compilation scheduler.
  mlir::MLIRContext *getContext() { return context; };

  /// @brief Compile only at the MLIR level for the full target
  /// system.
  /// @param moduleOp The root module operation to compile for.
  /// This must not be specialized to a system already.
  virtual llvm::Error compileMLIR(mlir::ModuleOp moduleOp) = 0;

  /// @brief Generate the full configured compilation pipeline
  /// for all targets of the base target system. This will also
  /// invoke compileMLIR.
  /// @param moduleOp The root module operation to compile for.
  /// This must not be specialized to a system already.
  /// @param payload The payload to populate.
  /// @param doCompileMLIR Whether to call compileMLIR prior to compiling the
  /// payload. Defaults to true.
  virtual llvm::Error compilePayload(mlir::ModuleOp moduleOp,
                                     qssc::payload::Payload &payload,
                                     bool doCompileMLIR = true) = 0;

  void enableIRPrinting(bool printBeforeAllTargetPasses,
                        bool printAfterAllTargetPasses,
                        bool printBeforeAllTargetPayload,
                        bool printAfterTargetCompileFailure);

protected:
  bool getPrintBeforeAllTargetPasses() { return printBeforeAllTargetPasses; }
  bool getPrintAfterAllTargetPasses() { return printAfterAllTargetPasses; }
  bool getPrintBeforeAllTargetPayload() { return printBeforeAllTargetPayload; }
  bool getPrintAfterTargetCompileFailure() {
    return printAfterTargetCompileFailure;
  }

  /// Thread-safe implementation
  virtual void printIR(llvm::StringRef msg, mlir::Operation *op,
               llvm::raw_ostream &out);

private:
  hal::TargetSystem &target;
  mlir::MLIRContext *context;

  bool printBeforeAllTargetPasses = false;
  bool printAfterAllTargetPasses = false;
  bool printBeforeAllTargetPayload = false;
  bool printAfterTargetCompileFailure = false;

}; // class TargetCompilationManager

/// Register a set of useful command-line options that can be used to configure
/// a target compilation scheduler.
void registerTargetCompilationManagerCLOptions();

/// Apply any values provided to the target compilation scheduler options that
/// were registered with 'registerTargetCompilationManagerCLOptions'.
mlir::LogicalResult
applyTargetCompilationManagerCLOptions(TargetCompilationManager &scheduler);

} // namespace qssc::hal::compile
#endif // TARGETCOMPILATIONMANAGER_H
