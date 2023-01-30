//===- SchedulePort.h  - Schedule Pulse on single port ----------*- C++ -*-===//
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
///  See SchedulePort.cpp for more detailed background.
//===----------------------------------------------------------------------===//

#ifndef PULSE_SCHEDULE_H
#define PULSE_SCHEDULE_H

#include "Dialect/Pulse/IR/PulseOps.h"

// #include <map>
// #include <vector>

namespace mlir::pulse {

using mixedFrameMap_t = std::map<uint, std::vector<Operation *>>;

mixedFrameMap_t buildMixedFrameMap(SequenceOp &sequenceOp,
                                   uint &numMixedFrames);

void sortOpsByTimepoint(SequenceOp &sequenceOp);

template <typename GetDuration>
void addTimepoints(mlir::OpBuilder &builder,
                   mixedFrameMap_t &mixedFrameSequences, uint &maxTime,
                   GetDuration getDuration) {

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
      if (auto castOp = dyn_cast<DelayOp>(op))
        currentTimepoint += castOp.getDuration();
      else if (auto castOp = dyn_cast<PlayOp>(op))
        currentTimepoint += getDuration(castOp);
    }
    if (currentTimepoint > maxTime)
      maxTime = currentTimepoint;
  }
} // addTimepoints

template <typename GetDuration>
uint processSequence(SequenceOp sequenceOp, GetDuration getDuration) {

  // TODO: Consider returning overall length of sequence to help schedule
  // across sequences
  mlir::OpBuilder builder(sequenceOp);

  uint numMixedFrames = 0;
  auto mixedFrameSequences = buildMixedFrameMap(sequenceOp, numMixedFrames);

  uint maxTime = 0;

  addTimepoints(builder, mixedFrameSequences, maxTime, getDuration);

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

} // namespace mlir::pulse

#endif // PULSE_SCHEDULE_CHANNEL_H
