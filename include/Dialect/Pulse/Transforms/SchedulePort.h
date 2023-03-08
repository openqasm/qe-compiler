//===- SchedulePort.h  - Schedule Pulse on single port ----------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
    : public PassWrapper<SchedulePortPass, OperationPass<ModuleOp>>,
      protected qssc::utils::DebugIndent {
public:
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;

private:
  using opVec_t = std::vector<Operation *>;
  using mixedFrameMap_t = std::map<uint, std::vector<Operation *>>;
  using opQueue_t = std::deque<Operation *>;

  std::deque<Operation *> removeList;
  uint processCall(Operation *module, CallSequenceOp &callSequenceOp);
  uint processCallee(Operation *module, CallSequenceOp &callSequenceOp,
                     Operation *findOp);

  mixedFrameMap_t buildMixedFrameMap(CallSequenceOp &callSequenceOp,
                                     SequenceOp &sequenceOp,
                                     uint &numMixedFrames);

  void addTimepoints(CallSequenceOp &callSequenceOp, mlir::OpBuilder &builder,
                     mixedFrameMap_t &mixedFrameSequences, uint &maxTime);

  void sortOpsByTimepoint(SequenceOp &sequenceOp);
  void removePendingOps();
};
} // namespace mlir::pulse

#endif // PULSE_SCHEDULE_CHANNEL_H
