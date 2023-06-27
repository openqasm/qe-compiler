//===- SimulatorUtils.h ----------------------------------------------*- C++ -*-===//
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
//
// Declaration of utility functions for the Simulator Target
//
//===----------------------------------------------------------------------===//

#ifndef HAL_SIMULATORUTILS_H
#define HAL_SIMULATORUTILS_H

#include <vector>

namespace mlir {
class ModuleOp;
} // end namespace mlir

namespace qssc::targets::simulator {

// Looks for and returns the Controller submodule if it exists
// Returns nullptr otherwise
auto getControllerModule(mlir::ModuleOp topModuleOp) -> mlir::ModuleOp;

auto getActuatorModules(mlir::ModuleOp topModuleOp)
    -> std::vector<mlir::ModuleOp>;

} // end namespace qssc::targets::simulator

#endif // HAL_SIMULATORUTILS_H
