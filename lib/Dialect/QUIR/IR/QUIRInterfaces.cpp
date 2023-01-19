//===- QUIRInterfaces.cpp - QUIR dialect interfaces ------------ *- C++ -*-===//
//
// (C) Copyright IBM 2022, 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file defines the QUIR dialect interfaces.
///
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/IR/QUIRInterfaces.h"

using namespace mlir::quir;

//===----------------------------------------------------------------------===//
// Tablegen Interface Definitions
//===----------------------------------------------------------------------===//

#include "Dialect/QUIR/IR/QUIRInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// QubitOpInterface
//===----------------------------------------------------------------------===//

std::set<uint32_t> interfaces_impl::getOperatedQubits(mlir::Operation *op,
                                                      bool ignoreSelf) {
  std::set<uint32_t> opQubits;
  op->walk([&](mlir::Operation *walkOp) {
    if (ignoreSelf && walkOp == op)
      return WalkResult::advance();
    if (QubitOpInterface interface = dyn_cast<QubitOpInterface>(walkOp)) {
      auto currOpQubits = interface.getOperatedQubits();
      opQubits.insert(currOpQubits.begin(), currOpQubits.end());
      // Avoid recursing again
      return WalkResult::skip();
    }
    return WalkResult::advance();
  });
  return opQubits;
}

llvm::Optional<mlir::Operation *>
interfaces_impl::getNextQubitOp(mlir::Operation *op) {
  mlir::Operation *curOp = op;
  while (mlir::Operation *nextOp = curOp->getNextNode()) {
    if (isa<QubitOpInterface>(nextOp))
      return nextOp;
    curOp = nextOp;
  }
  return llvm::None;
}

std::set<uint32_t>
interfaces_impl::getSharedQubits(std::set<uint32_t> &first,
                                 std::set<uint32_t> &second) {

  std::set<uint32_t> sharedQubits;
  std::set_intersection(first.begin(), first.end(), second.begin(),
                        second.end(),
                        std::inserter(sharedQubits, sharedQubits.begin()));

  return sharedQubits;
}

std::set<uint32_t> interfaces_impl::getUnionQubits(std::set<uint32_t> &first,
                                                   std::set<uint32_t> &second) {

  std::set<uint32_t> unionQubits;
  std::set_union(first.begin(), first.end(), second.begin(), second.end(),
                 std::inserter(unionQubits, unionQubits.begin()));

  return unionQubits;
}

bool interfaces_impl::qubitSetsOverlap(std::set<uint32_t> &first,
                                       std::set<uint32_t> &second) {
  std::set<uint32_t> sharedQubits = getSharedQubits(first, second);
  return !sharedQubits.empty();
}

std::set<uint32_t> interfaces_impl::getSharedQubits(Operation *first,
                                                    Operation *second) {
  auto leftQubits = getOperatedQubits(first);
  auto rightQubits = getOperatedQubits(second);

  return getSharedQubits(leftQubits, rightQubits);
}

bool interfaces_impl::opsShareQubits(Operation *first, Operation *second) {
  return !getSharedQubits(first, second).empty();
}

// TODO: A DAG should be used for this sort of analysis.
std::set<uint32_t>
interfaces_impl::getQubitsBetweenOperations(mlir::Operation *first,
                                            mlir::Operation *second) {
  std::set<uint32_t> operatedQubits;
  if (!first->isBeforeInBlock(second))
    return operatedQubits;

  Operation *curOp = first->getNextNode();
  while (Operation *nextOp = curOp->getNextNode()) {
    // Loop through qubits in block and find matching node.
    if (nextOp == second)
      return operatedQubits;
    auto nextOpQubits = getOperatedQubits(nextOp);
    operatedQubits.insert(nextOpQubits.begin(), nextOpQubits.end());
    curOp = nextOp;
  }
  return {};
}
