//===- SchedulePortSequence.cpp - Schedule Ops on single port ---*- C++ -*-===//
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
///  on the underlying port. For more detail see SchedulePort.cpp
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/Transforms/SchedulePortSequence.h"
#include "Dialect/Pulse/IR/PulseTypes.h"
#include "Dialect/Pulse/Utils/SchedulePort.h"
#include "Dialect/Pulse/Utils/Utils.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "SchedulePortSequencePass"

using namespace mlir;
using namespace mlir::pulse;

void SchedulePortSequencePass::runOnOperation() {

  INDENT_DEBUG("===== SchedulePortSequencePass - start ==========\n");

  processSequence(
      getOperation(), [&](PlayOp playOp) {
        if (!playOp->hasAttrOfType<IntegerAttr>("pulse.duration")) {
          playOp.emitError()
              << "SchedulingPortSequencePass requires that PlayOps be "
                 "labeled with a pulse.duration attribute";
          signalPassFailure();
        }
        return playOp->getAttrOfType<IntegerAttr>("pulse.duration").getUInt();
      });

  INDENT_DEBUG("=====  SchedulePortSequencePass - end ===========\n");

} // runOnOperation

llvm::StringRef SchedulePortSequencePass::getArgument() const {
  return "pulse-schedule-port";
}

llvm::StringRef SchedulePortSequencePass::getDescription() const {
  return "Schedule operations on the same port in a sequence";
}
