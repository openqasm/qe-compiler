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

#ifndef PULSE_SCHEDULE_PORT_H
#define PULSE_SCHEDULE_PORT_H

#include "Dialect/Pulse/IR/PulseOps.h"
#include "Utils/DebugIndent.h"
#include "mlir/Pass/Pass.h"

#include <deque>
#include <set>

namespace mlir::pulse {

class SchedulePortPass
    : public PassWrapper<SchedulePortPass, OperationPass<SequenceOp>>,
      protected qssc::utils::DebugIndent {
public:
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;

private:
  using opVec_t = std::vector<Operation *>;
  using mixedFrameMap_t = std::map<uint, std::vector<Operation *>>;
  using opQueue_t = std::deque<Operation *>;

  uint processCall(Operation *module, CallSequenceOp &callSequenceOp);
  uint processCallee(SequenceOp sequenceOp);

  mixedFrameMap_t buildMixedFrameMap(SequenceOp &sequenceOp,
                                     uint &numMixedFrames);

  void addTimepoints(mlir::OpBuilder &builder,
                     mixedFrameMap_t &mixedFrameSequences, uint &maxTime);

  void sortOpsByTimepoint(SequenceOp &sequenceOp);
};
} // namespace mlir::pulse

#endif // PULSE_SCHEDULE_CHANNEL_H
