//===- SchedulePort.cpp - Schedule Ops on single port -----------*- C++ -*-===//
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
///  This file implements the pass for scheduling on a single port.
///
///  A single port may have multiple frames mixed with it (measurement vs drive,
///  etc). Each mixed frame will have delay and play operations on the mixed
///  frame which need to be processed down to a set of delays and plays
///  on the underlying port.
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/Transforms/SchedulePort.h"
#include "Dialect/Pulse/IR/PulseInterfaces.h"
#include "Dialect/Pulse/IR/PulseOps.h"
#include "Dialect/Pulse/IR/PulseTraits.h"
#include "Dialect/Pulse/IR/PulseTypes.h"
#include "Dialect/Pulse/Utils/Utils.h"
#include "Utils/DebugIndent.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <utility>

#define DEBUG_TYPE "SchedulePortPass"

using namespace mlir;
using namespace mlir::pulse;

uint64_t SchedulePortPass::processCall(CallSequenceOp &callSequenceOp,
                                       bool updateNestedSequences) {

  INDENT_DEBUG("==== processCall - start  ===================\n");
  INDENT_DUMP(callSequenceOp.dump());
  INDENT_DEBUG("=============================================\n");

  // check for nested sequence
  auto parentSequence = callSequenceOp->getParentOfType<SequenceOp>();
  if (!updateNestedSequences && parentSequence)
    return 0;

  // walk into region and check arguments
  // look for sequence def match
  auto callee = callSequenceOp.getCallee();
  auto sequenceOpIter = sequenceOps.find(callee);

  if (sequenceOpIter == sequenceOps.end()) {
    callSequenceOp->emitError()
        << "Unable to find callee symbol " << callee << ".";
    signalPassFailure();
  }

  auto sequenceOp = sequenceOpIter->second;

  uint64_t calleeDuration;
  if (updateNestedSequences)
    calleeDuration = updateSequence(sequenceOp);
  else
    calleeDuration = processSequence(sequenceOp);
  PulseOpSchedulingInterface::setDuration(callSequenceOp, calleeDuration);

  INDENT_DEBUG("====  processCall - end  ====================\n");
  INDENT_DUMP(callSequenceOp.dump());
  INDENT_DEBUG("=============================================\n");
  return calleeDuration;
}

uint64_t SchedulePortPass::processSequence(SequenceOp sequenceOp) {

  mlir::OpBuilder builder(sequenceOp);

  uint32_t numMixedFrames = 0;
  auto mixedFrameSequences = buildMixedFrameMap(sequenceOp, numMixedFrames);

  int64_t maxTime = 0;

  addTimepoints(builder, mixedFrameSequences, maxTime);

  // remove all DelayOps - they are no longer required now that we have
  // timepoints
  sequenceOp->walk([&](DelayOp op) { op->erase(); });

  // sort updated ops so that ops across mixed frame are in the correct
  // sequence with respect to timepoint on a single port.
  sortOpsByTimepoint(sequenceOp);

  // clean up
  sequenceOp->walk([&](arith::ConstantOp op) {
    if (op->getUsers().empty())
      op->erase();
  });

  // assign timepoint to return
  // TODO: check for a better way to do this with getTerminator or back()
  sequenceOp->walk([&](ReturnOp op) {
    PulseOpSchedulingInterface::setTimepoint(op, maxTime);
  });
  return maxTime;
}

uint64_t SchedulePortPass::updateSequence(SequenceOp sequenceOp) {

  uint64_t updateDelta = 0;
  int64_t returnTimepoint = 0;
  for (Region &region : sequenceOp->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        int64_t timepoint = 0;

        auto existingTimepoint = PulseOpSchedulingInterface::getTimepoint(&op);
        if (existingTimepoint.has_value()) {
          timepoint = existingTimepoint.value() + updateDelta;
          PulseOpSchedulingInterface::setTimepoint(&op, timepoint);
        }

        if (auto castOp = dyn_cast<ReturnOp>(op)) {
          returnTimepoint = timepoint;
        } else if (auto castOp = dyn_cast<CallSequenceOp>(op)) {
          // a nested sequence should only have a duration if it has been
          // updated by this method already
          if (!castOp->hasAttr("pulse.duration")) {
            uint64_t const calleeDuration =
                processCall(castOp, /*updateNested*/ true);
            PulseOpSchedulingInterface::setDuration(castOp, calleeDuration);
            updateDelta += calleeDuration;
          } else {
            updateDelta +=
                castOp->getAttrOfType<IntegerAttr>("pulse.duration").getInt();
          }
        }
      }
    }
  }
  return returnTimepoint;
}

SchedulePortPass::mixedFrameMap_t
SchedulePortPass::buildMixedFrameMap(SequenceOp &sequenceOp,
                                     uint32_t &numMixedFrames) {

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
          if (auto castOp = dyn_cast<DelayOp>(op))
            target = castOp.getTarget();
          else if (auto castOp = dyn_cast<PlayOp>(op))
            target = castOp.getTarget();
          else if (auto castOp = dyn_cast<CaptureOp>(op))
            target = castOp.getTarget();
          else if (auto castOp = dyn_cast<SetFrequencyOp>(op))
            target = castOp.getTarget();
          else if (auto castOp = dyn_cast<SetPhaseOp>(op))
            target = castOp.getTarget();
          else if (auto castOp = dyn_cast<ShiftFrequencyOp>(op))
            target = castOp.getTarget();
          else if (auto castOp = dyn_cast<ShiftPhaseOp>(op))
            target = castOp.getTarget();
          else if (auto castOp = dyn_cast<SetAmplitudeOp>(op))
            target = castOp.getTarget();

          auto blockArg = target.cast<BlockArgument>();
          auto index = blockArg.getArgNumber();

          mixedFrameSequences[index].push_back(&op);
        } else if (auto castOp = dyn_cast<CallSequenceOp>(op)) {
          // add a call sequence to all mixedFrameSequences for
          // mixedFrames passed to it
          for (auto operand : castOp.getOperands()) {
            auto operandType = operand.getType();
            if (operandType.isa<MixedFrameType>()) {
              auto blockArg = operand.cast<BlockArgument>();
              auto index = blockArg.getArgNumber();
              mixedFrameSequences[index].push_back(&op);
            }
          }
        }
      }
    }
  }
  return mixedFrameSequences;
} // buildMixedFrameMap

void SchedulePortPass::addTimepoints(mlir::OpBuilder &builder,
                                     mixedFrameMap_t &mixedFrameSequences,
                                     int64_t &maxTime) {

  // add timepoint to operations in mixedFrameSequences where timepoints
  // are calculated based on the duration of delayOps
  //
  // Timepoints start at 0 for each mixed frame vector and are calculated
  // independently for each mixed frame.

  for (const auto &index : mixedFrameSequences) {
    int64_t currentTimepoint = 0;
    for (auto *op : index.second) {
      auto existingTimepoint = PulseOpSchedulingInterface::getTimepoint(op);
      if (existingTimepoint.has_value())
        if (existingTimepoint.value() > currentTimepoint)
          currentTimepoint = existingTimepoint.value();

      // set attribute on op with current timepoint
      PulseOpSchedulingInterface::setTimepoint(op, currentTimepoint);

      // update currentTimepoint if DelayOp or playOp
      if (auto delayOp = dyn_cast<DelayOp>(op)) {
        llvm::Expected<uint64_t> durOrError =
            PulseOpSchedulingInterface::getDuration<DelayOp>(delayOp);
        if (auto err = durOrError.takeError()) {
          delayOp.emitError() << toString(std::move(err));
          signalPassFailure();
        }
        currentTimepoint += durOrError.get();
      } else if (auto playOp = dyn_cast<PlayOp>(op)) {
        llvm::Expected<uint64_t> durOrError =
            playOp.getDuration(nullptr /*callSequenceOp*/);
        if (auto err = durOrError.takeError()) {
          playOp.emitError() << toString(std::move(err));
          signalPassFailure();
        }
        currentTimepoint += durOrError.get();
      }
    }
    if (currentTimepoint > maxTime)
      maxTime = currentTimepoint;
  }
} // addTimepoints

void SchedulePortPass::runOnOperation() {

  Operation *module = getOperation();

  module->walk(
      [&](mlir::pulse::SequenceOp op) { sequenceOps[op.getSymName()] = op; });

  INDENT_DEBUG("===== SchedulePortPass - start ==========\n");

  // assign
  module->walk([&](CallSequenceOp op) {
    processCall(op, /*updateNestedSequences*/ false);
  });
  module->walk([&](CallSequenceOp op) {
    processCall(op, /*updateNestedSequences*/ true);
  });

  INDENT_DEBUG("=====  SchedulePortPass - end ===========\n");

} // runOnOperation

llvm::StringRef SchedulePortPass::getArgument() const {
  return "pulse-schedule-port";
}

llvm::StringRef SchedulePortPass::getDescription() const {
  return "Schedule operations on the same port in a sequence";
}

llvm::StringRef SchedulePortPass::getName() const {
  return "Schedule Port Pass";
}
