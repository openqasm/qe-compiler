//===- SchedulePort.h  - Schedule Pulse on single port ----------*- C++ -*-===//
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
///  This file implements the pass for scheduling on a single port. The
///  pass operates at the module level.
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

#include <map>
#include <vector>

namespace mlir::pulse {

class SchedulePortPass
    : public PassWrapper<SchedulePortPass, OperationPass<ModuleOp>>,
      protected qssc::utils::DebugIndent {
public:
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
  llvm::StringRef getName() const override;

private:
  using mixedFrameMap_t = std::map<uint32_t, std::vector<Operation *>>;

  uint64_t processCall(CallSequenceOp &callSequenceOp,
                       bool updateNestedSequences);
  uint64_t processSequence(SequenceOp sequenceOp);
  uint64_t updateSequence(SequenceOp sequenceOp);

  mixedFrameMap_t buildMixedFrameMap(SequenceOp &sequenceOp,
                                     uint32_t &numMixedFrames);

  void addTimepoints(mlir::OpBuilder &builder,
                     mixedFrameMap_t &mixedFrameSequences, int64_t &maxTime);
  llvm::StringMap<mlir::pulse::SequenceOp> sequenceOps;
};
} // namespace mlir::pulse

#endif // PULSE_SCHEDULE_PORT_H
