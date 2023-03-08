//===- MergeDelays.h - Merge back to back pulse.delays  ---------*- C++ -*-===//
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
///  This file declares the pass for merging back to back pulse.delays.
///  The current implementation defaults to ignoring the target, there
///  is an option to override this.
///
//===----------------------------------------------------------------------===//

#ifndef PULSE_MERGE_DELAYS_H
#define PULSE_MERGE_DELAYS_H

#include "mlir/Pass/Pass.h"

namespace mlir::pulse {

class MergeDelayPass
    : public PassWrapper<MergeDelayPass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
};
} // namespace mlir::pulse

#endif // PULSE_MERGE_DELAY_H
