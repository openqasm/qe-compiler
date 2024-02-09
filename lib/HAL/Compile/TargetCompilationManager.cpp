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

#include "HAL/Compile/TargetCompilationManager.h"

#include "HAL/TargetSystem.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/Timing.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"

using namespace qssc::hal::compile;

namespace {
struct TargetCompilationManagerOptions {

  //===--------------------------------------------------------------------===//
  // IR Printing
  //===--------------------------------------------------------------------===//
  llvm::cl::opt<bool> printBeforeAllTargetPasses{
      "print-ir-before-all-target-passes",
      llvm::cl::desc("Print IR before each target compilation pass"),
      llvm::cl::init(false)};
  llvm::cl::opt<bool> printAfterAllTargetPasses{
      "print-ir-after-all-target-passes",
      llvm::cl::desc("Print IR after each target compilation pass"),
      llvm::cl::init(false)};
  llvm::cl::opt<bool> printBeforeAllTargetPayload{
      "print-ir-before-emit-all-target-payloads",
      llvm::cl::desc("Print IR before each target payload compilation"),
      llvm::cl::init(false)};
  llvm::cl::opt<bool> printAfterTargetCompileFailure{
      "print-ir-after-target-compile-failure",
      llvm::cl::desc("Print IR after failure of applying target compilation"),
      llvm::cl::init(false)};
};

llvm::ManagedStatic<TargetCompilationManagerOptions> options;

} // anonymous namespace

void qssc::hal::compile::registerTargetCompilationManagerCLOptions() {
  // Make sure that the options struct has been constructed.
  *options;
}

mlir::LogicalResult qssc::hal::compile::applyTargetCompilationManagerCLOptions(
    TargetCompilationManager &scheduler) {
  if (!options.isConstructed())
    return mlir::failure();

  // Otherwise, add the IR printing instrumentation.
  scheduler.enableIRPrinting(options->printBeforeAllTargetPasses,
                             options->printAfterAllTargetPasses,
                             options->printBeforeAllTargetPayload,
                             options->printAfterTargetCompileFailure);

  return mlir::success();
}

TargetCompilationManager::TargetCompilationManager(
    qssc::hal::TargetSystem &target, mlir::MLIRContext *context)
    : target(target), context(context) {}

llvm::Error TargetCompilationManager::walkTargetModules(
    Target *target, mlir::ModuleOp targetModuleOp, mlir::TimingScope &timing,
    const WalkTargetModulesFunction &walkFunc,
    const WalkTargetModulesFunction &postChildrenCallbackFunc) {
  // Call the input function for the walk on the target
  if (auto err = walkFunc(target, targetModuleOp, timing))
    return err;

  for (auto *child : target->getChildren()) {
    // Recurse on the target
    auto childModuleOp = child->getModule(targetModuleOp);
    if (auto err = childModuleOp.takeError())
      return err;
    if (auto err = walkTargetModules(child, *childModuleOp, timing, walkFunc,
                                     postChildrenCallbackFunc))
      return err;
  }

  if (auto err = postChildrenCallbackFunc(target, targetModuleOp, timing))
    return err;

  return llvm::Error::success();
}

llvm::Error
TargetCompilationManager::walkTarget(Target *target, mlir::TimingScope &timing,
                                     const WalkTargetFunction &walkFunc) {
  // Call the input function for the walk on the target
  if (auto err = walkFunc(target, timing))
    return err;

  for (auto *child : target->getChildren()) {
    // Recurse on the target
    if (auto err = walkTarget(child, timing, walkFunc))
      return err;
  }

  return llvm::Error::success();
}

// TODO: This should be replaced by a PassManager like instrumentation framework
void TargetCompilationManager::enableIRPrinting(
    bool printBeforeAllTargetPasses, bool printAfterAllTargetPasses,
    bool printBeforeAllTargetPayload, bool printAfterTargetCompileFailure) {
  this->printBeforeAllTargetPasses = printBeforeAllTargetPasses;
  this->printAfterAllTargetPasses = printAfterAllTargetPasses;
  this->printBeforeAllTargetPayload = printBeforeAllTargetPayload;
  this->printAfterTargetCompileFailure = printAfterTargetCompileFailure;
}

void TargetCompilationManager::printIR(llvm::Twine msg, mlir::Operation *op,
                                       llvm::raw_ostream &out) {
  out << "// -----// ";
  out << msg;
  out << " //----- //";
  out << "\n";
  mlir::OpPrintingFlags flags = mlir::OpPrintingFlags();
  op->print(out, flags.useLocalScope());
  out << "\n";
}

void TargetCompilationManager::enableTiming(mlir::TimingScope &timingScope) {
  rootTimer = timingScope.nest("TargetCompilationManager");
  rootTimer.hide();
}

void TargetCompilationManager::disableTiming() { rootTimer.stop(); }

mlir::TimingScope TargetCompilationManager::getTimer(llvm::StringRef name) {
  return rootTimer.nest(name);
}
