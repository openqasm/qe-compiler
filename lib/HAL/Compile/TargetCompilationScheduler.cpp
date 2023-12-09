//===- TargetCompilationScheduler.cpp ----------------------------*- C++
//-*-===//
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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ManagedStatic.h"

#include "HAL/Compile/TargetCompilationScheduler.h"

using namespace qssc::hal::compile;

namespace {
struct TargetCompilationSchedulerOptions {

  //===--------------------------------------------------------------------===//
  // IR Printing
  //===--------------------------------------------------------------------===//
  llvm::cl::opt<bool> printBeforeAll{
      "qssc-print-ir-before-all-targets", llvm::cl::desc("Print IR before each target compilation pass"),
      llvm::cl::init(false)};
  llvm::cl::opt<bool> printAfterAll{"qssc-print-ir-after-all-targets",
                                    llvm::cl::desc("Print IR after each target compilation pass"),
                                    llvm::cl::init(false)};
  llvm::cl::opt<bool> printAfterFailure{
      "qssc-print-ir-after-failure",
      llvm::cl::desc(
          "When printing the IR after a pass, only print if the pass failed"),
      llvm::cl::init(false)};
};
} // namespace


static llvm::ManagedStatic<TargetCompilationSchedulerOptions> options;

void qssc::hal::compile::registerTargetCompilationSchedulerCLOptions() {
  // Make sure that the options struct has been constructed.
  *options;
}

mlir::LogicalResult qssc::hal::compile::applyTargetCompilationSchedulerCLOptions(TargetCompilationScheduler &scheduler) {
  if (!options.isConstructed())
    return mlir::failure();

  // Otherwise, add the IR printing instrumentation.
  scheduler.enableIRPrinting(options->printBeforeAll, options->printAfterAll);

  return mlir::success();
}



TargetCompilationScheduler::TargetCompilationScheduler(
    qssc::hal::TargetSystem &target, mlir::MLIRContext *context)
    : target(target), context(context) {}

llvm::Error
TargetCompilationScheduler::walkTarget(Target *target,
                                       mlir::ModuleOp targetModuleOp,
                                       WalkTargetFunction walkFunc) {
  // Call the input function for the walk on the target
  if (auto err = walkFunc(target, targetModuleOp))
    return err;

  for (auto *child : target->getChildren()) {
    // Recurse on the target
    auto childModuleOp = child->getModule(targetModuleOp);
    if (auto err = childModuleOp.takeError())
      return err;
    if (auto err = walkTarget(child, *childModuleOp, walkFunc))
      return err;
  }
  return llvm::Error::success();
}

void TargetCompilationScheduler::enableIRPrinting(bool printBeforeAll, bool printAfterAll) {
  this->printBeforeAll = printBeforeAll;
  this->printAfterAll = printAfterAll;
}
