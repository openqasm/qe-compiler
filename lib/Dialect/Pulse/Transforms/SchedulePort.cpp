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
///  This file implements the pass for scheduling on a single port. The
///  pass operates at the module level. For an alternate pass which operates
///  at the sequence level see: SchedulePortSequence.{h,cpp}. Functionality
///  common to both passes is implemented in Utils/SchedulePort.{h,cpp}
///
///  A single port may have multiple frames mixed with it (measurement vs drive,
///  etc). Each mixed frame will have delay and play operations on the mixed
///  frame which need to be processed down to a set of delays and plays
///  on the underlying port. 
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/Transforms/SchedulePort.h"
#include "Dialect/Pulse/Utils/Utils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "SchedulePortPass"

using namespace mlir;
using namespace mlir::pulse;

uint64_t SchedulePortPass::processCall(Operation *module,
                                   CallSequenceOp &callSequenceOp) {

  INDENT_DEBUG("==== processCall - start  ===================\n");
  INDENT_DUMP(callSequenceOp.dump());
  INDENT_DEBUG("=============================================\n");

  // walk into region and check arguments
  // look for sequence def match
  auto callee = callSequenceOp.getCallee();
  auto sequenceOp =
      dyn_cast<SequenceOp>(SymbolTable::lookupSymbolIn(module, callee));
  uint64_t calleeDuration = processSequence(sequenceOp);
 

  INDENT_DEBUG("====  processCall - end  ====================\n");
  INDENT_DUMP(callSequenceOp.dump());
  INDENT_DEBUG("=============================================\n");
  return calleeDuration;
}

uint64_t SchedulePortPass::processSequence(SequenceOp sequenceOp) {

  // TODO: Consider returning overall length of sequence to help schedule
  // across sequences
  mlir::OpBuilder builder(sequenceOp);

  uint32_t numMixedFrames = 0;
  auto mixedFrameSequences = buildMixedFrameMap(sequenceOp, numMixedFrames);

  uint64_t maxTime = 0;

  addTimepoints(builder, mixedFrameSequences, maxTime);

  // remove all DelayOps - they are no longer required now that we have
  // timepoints
  sequenceOp->walk([&](DelayOp op) { op->erase(); });

  sortOpsByTimepoint(sequenceOp);

  // clean up
  sequenceOp->walk([&](arith::ConstantOp op) {
    if (op->getUsers().empty())
      op->erase();
  });

  // assign timepoint to return
  // TODO: check for a better way to do this with getTerminator or back()
  sequenceOp->walk([&](ReturnOp op) {
    IntegerAttr timepointAttr = builder.getI64IntegerAttr(maxTime);
    op->setAttr("pulse.timepoint", timepointAttr);
  });
  return maxTime;
}

void SchedulePortPass::runOnOperation() {

  Operation *module = getOperation();

  INDENT_DEBUG("===== SchedulePortPass - start ==========\n");

  module->walk([&](CallSequenceOp op) { processCall(module, op); });

  INDENT_DEBUG("=====  SchedulePortPass - end ===========\n");

} // runOnOperation

SchedulePortPass::mixedFrameMap_t 
SchedulePortPass::buildMixedFrameMap(SequenceOp &sequenceOp,
                                   uint &numMixedFrames) {

  // build a map between mixed frame (as represented by the arg index)
  // and a vector of operations on that mixed frame

  // process sequence arguments to initialize map with empty vectors
  mixedFrameMap_t mixedFrameSequences;
  for (auto const &argumentResult :
       llvm::enumerate(sequenceOp.getArguments())) {
    auto index = argumentResult.index();
    auto argumentType = argumentResult.value().getType();
    if (argumentType.isa<MixedFrameType>()) {
      numMixedFrames++;
      mixedFrameSequences[index] = {};
    }
  }

  // build vectors of operations on each mixed frame and push onto the
  // corresponding vector in the map
  //
  // currently only considering DelayOp and PlayOp
  for (Region &region : sequenceOp->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        if (op.hasTrait<mlir::pulse::HasTargetFrame>()) {
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

void  SchedulePortPass::addTimepoints(mlir::OpBuilder &builder,
                   mixedFrameMap_t &mixedFrameSequences, uint64_t &maxTime) {

  // add timepoint to operations in mixedFrameSequences where timepoints
  // are calculated based on the duration of delayOps
  //
  // Timepoints start at 0 for each mixed frame vector and are calculated
  // independently for each mixed frame.

  for (const auto &index : mixedFrameSequences) {
    uint currentTimepoint = 0;
    for (auto *op : index.second) {
      // set attribute on op with current timepoint
      IntegerAttr timepointAttr = builder.getI64IntegerAttr(currentTimepoint);
      op->setAttr("pulse.timepoint", timepointAttr);

      // update currentTimepoint if DelayOp or playOp
      if (auto delayOp = dyn_cast<DelayOp>(op))
        currentTimepoint += delayOp.getDuration();
      else if (auto playOp = dyn_cast<PlayOp>(op)) {
        if (!playOp->hasAttrOfType<IntegerAttr>("pulse.duration")) {
           playOp.emitError()
               << "SchedulingPortPass requires that PlayOps be "
                  "labeled with a pulse.duration attribute";
           signalPassFailure();
         }
        uint64_t duration = 
          playOp->getAttrOfType<IntegerAttr>("pulse.duration").getUInt();
        currentTimepoint += duration;
      }
    }
    if (currentTimepoint > maxTime)
      maxTime = currentTimepoint;
  }
} // addTimepoints

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

        if (!op1.hasTrait<mlir::pulse::HasTargetFrame>() ||
            !op2.hasTrait<mlir::pulse::HasTargetFrame>())
          return false;

        auto currentTime = getTimepoint(&op1);
        auto nextTime = getTimepoint(&op2);

        // order by timepoint
        return currentTime < nextTime;
      }); // blockOps.sort
    }
  }
} // sortOpsByType

llvm::StringRef SchedulePortPass::getArgument() const {
  return "pulse-schedule-port-module";
}

llvm::StringRef SchedulePortPass::getDescription() const {
  return "Schedule operations on the same port in a sequence";
}
