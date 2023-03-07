//===- MockUtils.cpp --------------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2023.
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
