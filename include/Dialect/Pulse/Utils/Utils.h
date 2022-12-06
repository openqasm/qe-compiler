//===- Utils.h - Pulse Utilities --------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
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

uint getQubitId(Port_CreateOp *pulsePortOp);

bool isPlayOpForDrive(Operation *op,
                      mlir::pulse::CallSequenceOp &callSequenceOp);

int getTimepoint(Operation *op);

static llvm::StringRef timepointAttrName = "timepoint";

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
