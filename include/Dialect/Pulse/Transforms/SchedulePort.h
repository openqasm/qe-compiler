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
///  pass operates at the module level. For an alternate pass which operates
///  at the sequence level see: SchedulePortSequence.{h,cpp}. Functionality
///  common to both passes is implemented in Utils/SchedulePort.{h,cpp}
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
  using mixedFrameMap_t = std::map<uint, std::vector<Operation *>>;
  uint64_t processCall(Operation *module, CallSequenceOp &callSequenceOp);
  mixedFrameMap_t buildMixedFrameMap(SequenceOp &sequenceOp,
                                   uint &numMixedFrames);
  void sortOpsByTimepoint(SequenceOp &sequenceOp);
  uint64_t processSequence(SequenceOp sequenceOp); 
  void addTimepoints(mlir::OpBuilder &builder,
                   mixedFrameMap_t &mixedFrameSequences, uint64_t &maxTime);                                 
};
} // namespace mlir::pulse

#endif // PULSE_SCHEDULE_PORT_H
