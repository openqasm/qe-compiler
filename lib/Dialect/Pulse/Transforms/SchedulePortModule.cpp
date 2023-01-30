//===- SchedulePortModule.cpp - Schedule Ops on single port -----*- C++ -*-===//
//
// (C) Copyright IBM 2022, 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file implements the pass for scheduling on a single port. The
///  pass operates at the module level. For an alternate pass which operates
///  at the sequence level see: SchedulePortSequence.{h,cpp}. Functionality
///  common to both passes is implemented in Utils/SchedulePort.{h,cpp}
///
///  A single port may have multiple frames mixed with it (measurement vs drive,
///  etc). Each mixed frame will have delay and play operations on the mixed
///  frame which need to be processed down to a set of delays and plays
///  on the underlying port. For more detail see SchedulePort.cpp
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/Transforms/SchedulePortModule.h"
#include "Dialect/Pulse/Utils/SchedulePort.h"
#include "Dialect/Pulse/Utils/Utils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "SchedulePortModulePass"

using namespace mlir;
using namespace mlir::pulse;

uint SchedulePortModulePass::processCall(Operation *module,
                                         CallSequenceOp &callSequenceOp) {

  INDENT_DEBUG("==== processCall - start  ===================\n");
  INDENT_DUMP(callSequenceOp.dump());
  INDENT_DEBUG("=============================================\n");

  // walk into region and check arguments
  // look for sequence def match
  auto callee = callSequenceOp.getCallee();
  auto sequenceOp =
      dyn_cast<SequenceOp>(SymbolTable::lookupSymbolIn(module, callee));
  uint calleeDuration = processSequence(sequenceOp, [&](PlayOp playOp) {
    auto duration = playOp.getDuration(callSequenceOp);
    if (auto err = duration.takeError()) {
      playOp.emitOpError() << toString(std::move(err));
      signalPassFailure();
    }
    return duration.get();
  });

  INDENT_DEBUG("====  processCall - end  ====================\n");
  INDENT_DUMP(callSequenceOp.dump());
  INDENT_DEBUG("=============================================\n");
  return calleeDuration;
}

void SchedulePortModulePass::runOnOperation() {

  Operation *module = getOperation();

  INDENT_DEBUG("===== SchedulePortModulePass - start ==========\n");

  module->walk([&](CallSequenceOp op) { processCall(module, op); });

  INDENT_DEBUG("=====  SchedulePortModulePass - end ===========\n");

} // runOnOperation

llvm::StringRef SchedulePortModulePass::getArgument() const {
  return "pulse-schedule-port-module";
}

llvm::StringRef SchedulePortModulePass::getDescription() const {
  return "Schedule operations on the same port in a sequence";
}
