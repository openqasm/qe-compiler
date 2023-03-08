//===- MockUtils.h ----------------------------------------------*- C++ -*-===//
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
//
// Declaration of utility functions for the Mock Target
//
//===----------------------------------------------------------------------===//

#ifndef HAL_MOCKUTILS_H
#define HAL_MOCKUTILS_H

#include <vector>

namespace mlir {
class ModuleOp;
} // end namespace mlir

namespace qssc::targets::mock {

// Looks for and returns the Controller submodule if it exists
// Returns nullptr otherwise
auto getControllerModule(mlir::ModuleOp topModuleOp) -> mlir::ModuleOp;

auto getActuatorModules(mlir::ModuleOp topModuleOp)
    -> std::vector<mlir::ModuleOp>;

} // end namespace qssc::targets::mock

#endif // HAL_MOCKUTILS_H
