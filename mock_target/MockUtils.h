//===- MockUtils.h ------------------------------------------------*- C++
//-*-===//
//
// (C) Copyright IBM 2021.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
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
