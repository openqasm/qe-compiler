//===- MockUtils.cpp --------------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
// Definition of utility functions for the Mock Target
//
//===----------------------------------------------------------------------===//

#include "MockUtils.h"

#include <mlir/IR/BuiltinOps.h>

using namespace qssc::targets::mock;
using namespace mlir;

auto qssc::targets::mock::getControllerModule(ModuleOp topModuleOp)
    -> ModuleOp {
  ModuleOp retOp = nullptr;
  topModuleOp->walk([&](ModuleOp walkOp) {
    auto nodeType = walkOp->getAttrOfType<StringAttr>("quir.nodeType");
    if (nodeType && nodeType.getValue() == "controller") {
      retOp = walkOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return retOp;
}

auto qssc::targets::mock::getActuatorModules(ModuleOp topModuleOp)
    -> std::vector<ModuleOp> {
  std::vector<ModuleOp> retVec;
  topModuleOp->walk([&](ModuleOp walkOp) {
    auto nodeType = walkOp->getAttrOfType<StringAttr>("quir.nodeType");
    if (nodeType &&
        (nodeType.getValue() == "drive" || nodeType.getValue() == "acquire"))
      retVec.emplace_back(walkOp);
  });
  return retVec;
}
