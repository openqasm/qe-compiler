//===- Utils.h - Pulse Utilities --------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022, 2023.
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
///  This file declares some utility functions for Pulse passes
///
//===----------------------------------------------------------------------===//

#include "Dialect/Pulse/IR/PulseOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/StringRef.h"

#include <vector>

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::pulse {

Waveform_CreateOp getWaveformOp(PlayOp pulsePlayOp,
                                CallSequenceOp callSequenceOp);

template <typename PulseOpTy>
MixFrameOp getMixFrameOp(PulseOpTy pulseOp, CallSequenceOp callSequenceOp) {

  auto frameArgIndex =
      pulseOp.target().template cast<BlockArgument>().getArgNumber();
  auto frameOp = callSequenceOp.getOperand(frameArgIndex).getDefiningOp();

  auto mixFrameOp = dyn_cast<mlir::pulse::MixFrameOp>(frameOp);

  if (!mixFrameOp)
    pulseOp->emitError() << "The target argument is not a MixFrameOp.";
  return mixFrameOp;
}

uint getQubitId(MixFrameOp mixFrameOp);
uint getQubitId(Port_CreateOp pulsePortOp);

bool isPlayOpForDrive(Operation *op,
                      mlir::pulse::CallSequenceOp callSequenceOp);

int getTimepoint(Operation *op);

template <typename Func>
void iterateByTimepoint(std::vector<Operation *> &ops, Func func) {
  // template function for iterating over a list of operations by timepoint
  // the ops vector is sorted and then iterated over skipping timepoints
  // func is passed the currentOp, the Op with the next timepoint, and the
  // timepoint of the currentOp

  // sort ops by timepoint
  std::sort(ops.begin(), ops.end(), [&](Operation *a, Operation *b) {
    return getTimepoint(a) < getTimepoint(b);
  });

  // iterate over the sorted vector
  auto currentOp = ops.begin();
  while (currentOp != ops.end()) {
    auto currentTimepoint = getTimepoint(*currentOp);

    // get next op which has a timepoint greater than the current op
    auto nextOp = std::next(currentOp);
    while (nextOp != ops.end()) {
      if (getTimepoint(*nextOp) > currentTimepoint)
        break;
      nextOp = std::next(nextOp);
    }

    // call templated function
    func(currentOp, nextOp, currentTimepoint);

    // advance to next op
    currentOp = nextOp;
  }
} // iterateByTimepoint

} // end namespace mlir::pulse
