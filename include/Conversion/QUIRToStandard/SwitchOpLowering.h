//===- SwitchOpLowering.h ---------------------------------------*- C++ -*-===//
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
/// \file
/// This file exposes patterns for lowering quir.switch.
///
//===----------------------------------------------------------------------===//

#ifndef QUIRTOSTD_SWITCHOPLOWERING_H
#define QUIRTOSTD_SWITCHOPLOWERING_H

namespace mlir::quir {

void populateSwitchOpLoweringPatterns(RewritePatternSet &patterns);

}; // namespace mlir::quir

#endif // QUIRTOSTD_SWITCHOPLOWERING_H
