//===- TargetSystem.cpp -----------------------------------------*- C++ -*-===//
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

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallString.h"

#include "Dialect/QUIR/Utils/Utils.h"
#include "HAL/TargetSystem.h"

using namespace qssc::hal;
using namespace qssc::payload;

Target::Target(std::string name, Target *parent)
    : name(std::move(name)), parent(parent) {}

llvm::Error Target::emitToPayloadPostChildren(mlir::ModuleOp &targetModuleOp,
                                    payload::Payload &payload) { return llvm::Error::success(); }

TargetSystem::TargetSystem(std::string name, Target *parent)
    : Target(std::move(name), parent) {}

llvm::Expected<mlir::ModuleOp>
TargetSystem::getModule(mlir::ModuleOp parentModuleOp) {
  // For the system we treat the top-level parent module as the system module
  // currently.
  // TODO: Add a more general target module formalism
  return parentModuleOp;
}

llvm::Expected<TargetInstrument*> TargetSystem::getInstrumentWithNodeId(uint nodeId) const {
  for (auto *inst : getInstruments())
    if (inst->getNodeId() == nodeId)
      return inst;

  return llvm::createStringError(
    llvm::inconvertibleErrorCode(),
    "Could not find instrument with nodeId " + std::to_string(nodeId));

}

TargetInstrument::TargetInstrument(std::string name, Target *parent)
    : Target(std::move(name), parent) {}

llvm::Expected<mlir::ModuleOp>
TargetInstrument::getModule(mlir::ModuleOp parentModuleOp) {
  mlir::ModuleOp retOp = nullptr;
  parentModuleOp->walk([&](mlir::ModuleOp walkOp) {
    auto moduleNodeType =
        walkOp->getAttrOfType<mlir::StringAttr>("quir.nodeType");
    auto moduleNodeId = mlir::quir::getNodeId(walkOp);
    if (moduleNodeId.takeError())
      return mlir::WalkResult::advance();
    // Match by node type & id
    if (moduleNodeType && moduleNodeType.getValue() == getNodeType() &&
        moduleNodeId.get() == getNodeId()) {
      retOp = walkOp;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  if (!retOp) {
    llvm::outs() << "Couldn't find target \n";
    parentModuleOp.dump();
  }
  if (!retOp)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Could not find target module for target " + getName() +
            ". Searching for quir.nodeType=" + getNodeType() +
            " and quir.nodeId=" + std::to_string(getNodeId()));
  return retOp;
}
