//===- SchedulePort.cpp - Schedule Ops on single port -----------*- C++ -*-===//
//
// (C) Copyright IBM 2022, 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file implements the pass for scheduling on a single port
///
///  A single port may have multiple frames mixed with it (measurement vs drive,
///  etc). Each mixed frame will have delay and play operations on the mixed
///  frame which need to be processed down to a set of delays and plays
///  on the underlying port.
///
//===----------------------------------------------------------------------===//
//  For example:
//                    _____
//  Frame A:     _____|    |__________   , delay(2,a), play(w1,a), delay(13,a)
//               0    2    5          15
//
//                      _______
//  Frame B:     _______|      |______   , delay(7,b), play(w2,b), delay(8,b)
//              0       7      12     15
//
//  are processed to:
//                    _____
//  Frame A:     _____|    |             , delay(2,a), play(w1,a)
//               0    2    5
//
//                      _______
//  Frame B:          __|      |______   , delay(5,b), play(w2,b), delay(8,b)
//                    2 7      12     15
//
//  where the first delay on Frame B has been shortened to account for the
//  first delay on Frame A, and the second delay on Frame A has been removed
//  to account for the play and delay on Frame B.
//
//  This is accomplished by assigning timepoints to the delay and play
//  operations. Timepoints are assigned to each frame independently
//  as though they are playing concurrently.
//
//  The ops are then sorted by timepoint and delay ops are erased.
//
//  Pass assumptions:
//     Pass processes individual drive and acquire modules where those modules
//     only contain operations on the ports that the module is responsible
//     for. For example if the module is responsible for port 0 only operations
//     on port 0 are present in the module.
//
//     The pass assumes that the Pulse Ops to be processed are contained
//     within pulse.sequences.
//
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
