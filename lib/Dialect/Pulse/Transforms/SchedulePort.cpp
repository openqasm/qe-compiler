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

#include "Dialect/Pulse/Transforms/SchedulePort.h"
#include "Dialect/Pulse/Utils/Utils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "SchedulePortPass"

using namespace mlir;
using namespace mlir::pulse;

uint SchedulePortPass::processCall(Operation *module,
                                   CallSequenceOp &callSequenceOp) {

  INDENT_DEBUG("==== processCall - start  ===================\n");
  INDENT_DUMP(callSequenceOp.dump());
  INDENT_DEBUG("=============================================\n");

  // walk into region and check arguments
  // look for sequence def match
  auto callee = callSequenceOp.getCallee();
  Operation *findOp = SymbolTable::lookupSymbolIn(module, callee);
  uint calleeDuration = processCallee(module, callSequenceOp, findOp);

  INDENT_DEBUG("====  processCall - end  ====================\n");
  INDENT_DUMP(callSequenceOp.dump());
  INDENT_DEBUG("=============================================\n");
  return calleeDuration;
}

uint SchedulePortPass::processCallee(Operation *module,
                                     CallSequenceOp &callSequenceOp,
                                     Operation *findOp) {

  // TODO: Consider returning overall length of sequence to help schedule
  // across sequences

  auto sequenceOp = dyn_cast<SequenceOp>(findOp);
  if (!sequenceOp)
    return 0;

  mlir::OpBuilder builder(sequenceOp);

  uint numMixedFrames = 0;
  auto mixedFrameSequences =
      buildMixedFrameMap(callSequenceOp, sequenceOp, numMixedFrames);

  if (numMixedFrames < 2) {
    // if there is less than 2 mixed frames in this sequence then there is
    // no reason to change the schedule of the sequence
    return 0;
  }

  uint maxTime = 0;

  addTimepoints(callSequenceOp, builder, mixedFrameSequences, maxTime);

  // remove all DelayOps - they are no longer required now that we have
  // timepoints
  sequenceOp->walk([&](DelayOp op) { op->erase(); });

  sortOpsByTimepoint(sequenceOp);

  // clean up
  sequenceOp->walk([&](arith::ConstantOp op) {
    if (op->getUsers().empty())
      removeList.push_back(op);
  });

  removePendingOps();

  // assign timepoint to return
  // TODO: check for a better way to do this with getTerminator or back()
  sequenceOp->walk([&](ReturnOp op) {
    IntegerAttr timepointAttr = builder.getI64IntegerAttr(maxTime);
    op->setAttr("pulse.timepoint", timepointAttr);
  });

  INDENT_DEBUG("==== processCallee - end ===============\n");
  INDENT_DUMP(sequenceOp.dump());
  INDENT_DEBUG("===========================================\n");
  return maxTime;
}

SchedulePortPass::mixedFrameMap_t
SchedulePortPass::buildMixedFrameMap(CallSequenceOp &callSequenceOp,
                                     SequenceOp &sequenceOp,
                                     uint &numMixedFrames) {

  // build a map between mixed frame (as represented by the arg index)
  // and a vector of operations on that mixed frame

  // process sequence arguments to initialize map with empty vectors
  mixedFrameMap_t mixedFrameSequences;
  for (auto const &argumentResult :
       llvm::enumerate(sequenceOp.getArguments())) {
    auto index = argumentResult.index();
    auto *definingOp = callSequenceOp.getOperand(index).getDefiningOp();
    if (isa<MixFrameOp>(definingOp)) {
      numMixedFrames++;
      mixedFrameSequences[index] = {};
    }
  }
  if (numMixedFrames < 2) {
    // if there is only one mixed frame on a port, we don't have to interleave
    // the operations from multiple mixed frames that are being mapped to that
    // port. Basically, there will be no possible timing conflict, and therefor
    // we can return early
    return mixedFrameSequences;
  }

  // build vectors of operations on each mixed frame and push onto the
  // corresponding vector in the map
  //
  // currently only considering DelayOp and PlayOp
  for (Region &region : sequenceOp->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        if (op.hasTrait<mlir::pulse::HasTarget>()) {
          Value target;
          // get mixed_frames
          if (isa<DelayOp>(op))
            target = dyn_cast<DelayOp>(op).target();
          else if (isa<PlayOp>(op))
            target = dyn_cast<PlayOp>(op).target();
          else if (isa<CaptureOp>(op))
            target = dyn_cast<CaptureOp>(op).target();
          else if (isa<SetFrequencyOp>(op))
            target = dyn_cast<SetFrequencyOp>(op).target();
          else if (isa<SetPhaseOp>(op))
            target = dyn_cast<SetPhaseOp>(op).target();
          else if (isa<ShiftFrequencyOp>(op))
            target = dyn_cast<ShiftFrequencyOp>(op).target();
          else if (isa<ShiftPhaseOp>(op))
            target = dyn_cast<ShiftPhaseOp>(op).target();
          else if (isa<SetAmplitudeOp>(op))
            target = dyn_cast<SetAmplitudeOp>(op).target();

          auto blockArg = target.cast<BlockArgument>();
          auto index = blockArg.getArgNumber();

          mixedFrameSequences[index].push_back(&op);
        }
      }
    }
  }
  return mixedFrameSequences;
} // buildMixedFrameMap

void SchedulePortPass::addTimepoints(
    CallSequenceOp &callSequenceOp, mlir::OpBuilder &builder,
    mixedFrameMap_t &mixedFrameSequences, uint &maxTime) {

  // add timepoint to operations in mixedFrameSequences where timepoints
  // are calculated based on the duration of delayOps
  //
  // Timepoints start at 0 for each mixed frame vector and are calculated
  // independently for each mixed frame.

  for (const auto &index : mixedFrameSequences) {
    INDENT_DEBUG("processing fixed frame: " << index.first << "\n");
    increaseDebugIndent();

    uint currentTimepoint = 0;
    for (auto *op : index.second) {
      INDENT_DEBUG("current timepoint " << currentTimepoint);
      LLVM_DEBUG(llvm::errs() << " max_timepoint " << maxTime);
      LLVM_DEBUG(llvm::errs() << " op = ");
      LLVM_DEBUG(op->dump());

      // set attribute on op with current timepoint
      IntegerAttr timepointAttr = builder.getI64IntegerAttr(currentTimepoint);
      op->setAttr("pulse.timepoint", timepointAttr);

      // get duration of current op and add to current timepoint
      auto delayOp = dyn_cast<DelayOp>(op);
      if (delayOp)
        currentTimepoint += delayOp.getDuration();

      auto playOp = dyn_cast<PlayOp>(op);
      if (playOp) {
        auto duration = playOp.getDuration(callSequenceOp);
         if (auto err = duration.takeError()) {
           playOp.emitOpError() << toString(std::move(err));
           signalPassFailure();
         }
         currentTimepoint += duration.get();
      }  
    }
    if (currentTimepoint > maxTime)
      maxTime = currentTimepoint;
    decreaseDebugIndent();
  }
} // buildOpsList

void SchedulePortPass::sortOpsByTimepoint(SequenceOp &sequenceOp) {
  // sort updated ops so that ops across mixed frame are in the correct
  // sequence with respect to timepoint on a single port.

  // sort ops by timepoint
  for (Region &region : sequenceOp->getRegions()) {
    for (Block &block : region.getBlocks()) {
      auto &blockOps = block.getOperations();
       blockOps.sort([](Operation &op1, Operation &op2) {
         // put constants ahead of everything else
         if (isa<arith::ConstantIntOp>(op1) && !isa<arith::ConstantIntOp>(op2))
           return true;

         if (!op1.hasTrait<mlir::pulse::HasTarget>() || !op2.hasTrait<mlir::pulse::HasTarget>())
          return false;

         auto currentTime = getTimepoint(&op1);
         auto nextTime = getTimepoint(&op2);

         // order by timepoint
         if (currentTime < nextTime)
           return true;
         return false;
       }); // blockOps.sort
    }
  }
} // sortOpsByType

void SchedulePortPass::runOnOperation() {

  Operation *module = getOperation();

  INDENT_DEBUG("===== SchedulePortPass - start ==========\n");

  removeList.clear();

  module->walk([&](CallSequenceOp op) { processCall(module, op); });

  INDENT_DEBUG("=====  SchedulePortPass - end ===========\n");

} // runOnOperation

void SchedulePortPass::removePendingOps() {
  // remove any ops that were scheduled to be removed above.
  while (!removeList.empty()) {
    auto *op = removeList.front();
    INDENT_DEBUG("Removing ");
    LLVM_DEBUG(op->dump());
    removeList.pop_front();
    op->erase();
  }
}

llvm::StringRef SchedulePortPass::getArgument() const {
  return "pulse-schedule-port";
}

llvm::StringRef SchedulePortPass::getDescription() const {
  return "Schedule operations on the same port in a sequence";
}
