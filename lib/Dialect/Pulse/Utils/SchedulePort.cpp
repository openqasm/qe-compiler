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
///  This file implements the common functions for the SchedulePortModule
///  and SchedulePortSequence passes.
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

#include "Dialect/Pulse/Utils/SchedulePort.h"
#include "Dialect/Pulse/IR/PulseTypes.h"
#include "Dialect/Pulse/Utils/Utils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "SchedulePort"

using namespace mlir;

namespace mlir::pulse {
mixedFrameMap_t buildMixedFrameMap(SequenceOp &sequenceOp,
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

void sortOpsByTimepoint(SequenceOp &sequenceOp) {
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
} // namespace mlir::pulse
