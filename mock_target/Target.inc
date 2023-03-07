//===- Target.inc - Mock target registration --------------------*- C++ -*-===//
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
//
//  This file defines static objects that register system targets
//  with the QSS compiler core.
//
//===----------------------------------------------------------------------===//

#ifndef HAL_TARGETS_MOCK_TARGET_H
#define HAL_TARGETS_MOCK_TARGET_H

namespace qssc::targets::mock {

int init();
[[maybe_unused]] int registrar = init();

} // namespace qssc::targets::mock

#endif // HAL_TARGETS_MOCK_TARGET_H
