//===- SchedulePortSequence.h  - Schedule Pulse on single port ---- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file implements the pass for scheduling on a single port. The
///  pass operates at the sequence level. For an alternate pass which operates
///  at the module level see: SchedulePortModule.{h,cpp}. Functionality
///  common to both passes is implemented in Utils/SchedulePort.{h,cpp}
///
///  A single port may have multiple frames mixed with it (measurement vs drive,
///  etc). Each mixed frame will have delay and play operations on the mixed
///  frame which need to be processed down to a set of delays and plays
///  on the underlying port.
///
///  See SchedulePort.cpp for more detailed background.
//===----------------------------------------------------------------------===//

#ifndef PULSE_SCHEDULE_PORT_SEQUENCE_H
#define PULSE_SCHEDULE_PORT_SEQUENCE_H

#include "Dialect/Pulse/IR/PulseOps.h"
#include "Utils/DebugIndent.h"
#include "mlir/Pass/Pass.h"

#include <map>
#include <vector>

namespace mlir::pulse {

class SchedulePortSequencePass
    : public PassWrapper<SchedulePortSequencePass, OperationPass<SequenceOp>>,
      protected qssc::utils::DebugIndent {
public:
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;

private:
  using mixedFrameMap_t = std::map<uint, std::vector<Operation *>>;

  // uint processSequence(SequenceOp sequenceOp);

  // void addTimepoints(mlir::OpBuilder &builder,
  //                    mixedFrameMap_t &mixedFrameSequences, uint &maxTime);
};
} // namespace mlir::pulse

#endif // PULSE_SCHEDULE_CHANNEL_H
