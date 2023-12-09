//===- TargetCompilationScheduler.h - Compilation Scheduler -----*- C++ -*-===//
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
//
//  This file declares the classes for the top-level compilation scheduling
//  interface.
//
//===----------------------------------------------------------------------===//
#ifndef TARGETCOMPILATIONSCHEDULER_H
#define TARGETCOMPILATIONSCHEDULER_H

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
class TargetCompilationScheduler {
protected:
  TargetCompilationScheduler(hal::TargetSystem &target,
                             mlir::MLIRContext *context);

  using WalkTargetFunction =
      std::function<llvm::Error(hal::Target *, mlir::ModuleOp)>;
  // Depth first walker for a target system
  llvm::Error walkTarget(Target *target, mlir::ModuleOp targetModuleOp,
                         WalkTargetFunction walkFunc);

public:
  virtual ~TargetCompilationScheduler() = default;
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
  virtual llvm::Error compilePayload(mlir::ModuleOp moduleOp,
                                     qssc::payload::Payload &payload) = 0;

  void enableIRPrinting(bool printBeforeAll, bool printAfterAll);

private:
  bool getPrintBeforeAll() { return printBeforeAll; }
  bool getPrintAfterAll() { return printAfterAll; }

  hal::TargetSystem &target;
  mlir::MLIRContext *context;

  bool printBeforeAll = false;
  bool printAfterAll = false;

}; // class TargetCompilationScheduler


/// Register a set of useful command-line options that can be used to configure
/// a target compilation scheduler.
void registerTargetCompilationSchedulerCLOptions();

/// Apply any values provided to the target compilation scheduler options that were registered
/// with 'registerTargetCompilationSchedulerCLOptions'.
mlir::LogicalResult applyTargetCompilationSchedulerCLOptions(TargetCompilationScheduler &scheduler);

} // namespace qssc::hal::compile
#endif // TARGETCOMPILATIONSCHEDULER_H
