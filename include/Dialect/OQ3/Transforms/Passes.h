//===- Passes.h - OQ3 Passes  -----------------------------------*- C++ -*-===//
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

#ifndef OQ3_OQ3PASSES_H
#define OQ3_OQ3PASSES_H

#include "LimitCBitWidth.h"

namespace mlir::oq3 {
void registerOQ3Passes();
void registerOQ3PassPipeline();

} // namespace mlir::oq3

#endif // OQ3_OQ3PASSES_H
