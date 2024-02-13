//===- QubitLocalization.cpp - Create modules for qubit control -*- C++ -*-===//
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
//  This file implements the pass for creating localized qubit modules
//
//===----------------------------------------------------------------------===//

#include "QubitLocalization.h"

#include "MockTarget.h"

#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/QCS/IR/QCSOps.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <deque>
#include <memory>
#include <string>
#include <sys/types.h>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::quir;
using namespace mlir::qcs;
using namespace mlir::oq3;
namespace mock = qssc::targets::systems::mock;
using namespace mock;

namespace {
/// Returns true of this op has the classicalOnly attribute set to true
auto classicalOnlyCheck(Operation *op) -> bool {
  auto classicalOnlyAttr = op->getAttrOfType<BoolAttr>("quir.classicalOnly");
  if (classicalOnlyAttr)
    return classicalOnlyAttr.getValue();
  op->emitOpError() << "Error! No classicalOnly attribute found!\n"
                    << "Try running --classical-only-detection first!\n";
  return false;
}
} // end anonymous namespace

// find a qubit ID from the attribute on its declaration
auto mock::MockQubitLocalizationPass::lookupQubitId(const Value &val) -> int {
  auto declOp = val.getDefiningOp<DeclareQubitOp>();
  if (!declOp) { // Must be an argument to a function
    // see if we can find an attribute with the info
    if (auto blockArg = val.dyn_cast<BlockArgument>()) {
      unsigned const argIdx = blockArg.getArgNumber();
      auto funcOp =
          dyn_cast<mlir::func::FuncOp>(blockArg.getOwner()->getParentOp());
      if (funcOp) {
        auto argAttr =
            funcOp.getArgAttrOfType<IntegerAttr>(argIdx, "quir.physicalId");
        if (argAttr)
          return argAttr.getInt();
      } // if parentOp is funcOp
    }   // if val is blockArg
    return -1;
  } // if !declOp

  auto id = declOp.getId();
  if (id.has_value())
    return id.value();

  return -1;
} // lookupQubitId

/// Creates a broadcast op on Controller and recvOp on all other mocks
/// also checks if a value *should* be broadcast
void mock::MockQubitLocalizationPass::broadcastAndReceiveValue(
    const Value &val, const Location &loc,
    const std::unordered_set<uint> &toNodeIds) {
  if (alreadyBroadcastValues.count(val) == 0) {
    Operation *parentOp = val.getDefiningOp();
    if (parentOp) {
      if (dyn_cast<mlir::arith::ConstantOp>(parentOp) ||
          dyn_cast<quir::ConstantOp>(parentOp)) {
        // Just clone this op to the mocks
        for (uint const id : toNodeIds)
          (*mockBuilders)[id]->clone(*parentOp, mockMapping[id]);
      } else {
        controllerBuilder->create<BroadcastOp>(
            loc, controllerMapping.lookupOrNull(val));
        for (uint const id : toNodeIds) {
          auto recvOp = (*mockBuilders)[id]->create<RecvOp>(
              loc, TypeRange(val.getType()),
              controllerBuilder->getIndexArrayAttr(config->controllerNode()));
          mockMapping[id].map(val, recvOp.getVals().front());
        }
      }
      alreadyBroadcastValues.insert(val);
    }
    // else no parentOp means it's a block argument,
    // must have already been sent for the call, do nothing
  } // if alreadyBroadcastValues.count(val) == 0
} // broadcastValue

void mock::MockQubitLocalizationPass::cloneRegionWithoutOps(Region *from,
                                                            Region *dest,
                                                            IRMapping &mapper) {
  assert(dest && "expected valid region to clone into");
  cloneRegionWithoutOps(from, dest, dest->end(), mapper);
} // cloneRegionWithoutOps

// clone region (from) into region (dest) before the given position
void mock::MockQubitLocalizationPass::cloneRegionWithoutOps(
    Region *from, Region *dest, Region::iterator destPos, IRMapping &mapper) {
  assert(dest && "expected valid region to clone into");
  assert(from != dest && "cannot clone region into itself");

  // If the list is empty there is nothing to clone.
  if (from->empty())
    return;

  for (Block &block : from->getBlocks()) {
    auto *newBlock = new Block();
    mapper.map(&block, newBlock);

    // Clone the block arguments. The user might be deleting arguments to the
    // block by specifying them in the mapper. If so, we don't add the
    // argument to the cloned block.
    for (auto arg : block.getArguments())
      if (!mapper.contains(arg))
        mapper.map(arg, newBlock->addArgument(arg.getType(), arg.getLoc()));

    dest->getBlocks().insert(destPos, newBlock);
  }

  // Now that each of the blocks have been cloned, go through and remap the
  // operands of each of the operations.
  auto remapOperands = [&](Operation *op) {
    for (auto &operand : op->getOpOperands())
      if (auto mappedOp = mapper.lookupOrNull(operand.get()))
        operand.set(mappedOp);
    for (auto &succOp : op->getBlockOperands())
      if (auto *mappedOp = mapper.lookupOrNull(succOp.get()))
        succOp.set(mappedOp);
  };

  for (Region::iterator it(mapper.lookup(&from->front())); it != destPos; ++it)
    it->walk(remapOperands);
} // cloneRegionWithoutOps

auto mock::MockQubitLocalizationPass::addMainFunction(
    Operation *moduleOperation, const Location &loc) -> mlir::func::FuncOp {
  OpBuilder b(moduleOperation->getRegion(0));
  auto funcOp = b.create<mlir::func::FuncOp>(
      loc, "main",
      b.getFunctionType(
          /*inputs=*/ArrayRef<Type>(),
          /*results=*/ArrayRef<Type>(b.getI32Type())));
  funcOp.getOperation()->setAttr(llvm::StringRef("quir.classicalOnly"),
                                 b.getBoolAttr(false));
  funcOp.addEntryBlock();
  return funcOp;
} // addMainFunction

void mock::MockQubitLocalizationPass::processOp(DeclareQubitOp &qubitOp) {
  Operation *op = qubitOp.getOperation();
  llvm::outs() << "Localizing a " << op->getName() << "\n";

  // declare every qubit on each mock for multi-qubit gates purposes
  for (auto nodeId : seenNodeIds) {
    auto *clonedOp = (*mockBuilders)[nodeId]->clone(*op);
    mockMapping[nodeId].map(qubitOp.getRes(),
                            dyn_cast<DeclareQubitOp>(clonedOp).getRes());
  }
} // processOp DeclareQubitOp

void mock::MockQubitLocalizationPass::processOp(ResetQubitOp &resetOp) {
  if (resetOp.getQubits().size() != 1) {
    signalPassFailure(); // only support single-qubit resets"
    return;
  }

  Operation *op = resetOp.getOperation();
  llvm::outs() << "Localizing a " << op->getName() << "\n";
  int const qubitId = lookupQubitId(resetOp.getQubits().front());
  if (qubitId < 0) {
    resetOp->emitOpError() << "Can't resolve qubit ID for resetOp\n";
    return signalPassFailure();
  }
  (*mockBuilders)[config->driveNode(qubitId)]->clone(
      *op, mockMapping[config->driveNode(qubitId)]);
  (*mockBuilders)[config->acquireNode(qubitId)]->clone(
      *op, mockMapping[config->acquireNode(qubitId)]);
} // processOp ResetQubitOp

void mock::MockQubitLocalizationPass::processOp(mlir::func::FuncOp &funcOp) {
  Operation *op = funcOp.getOperation();
  OpBuilder const fBuild(funcOp);
  llvm::outs() << "Cloning FuncOp " << SymbolRefAttr::get(funcOp)
               << " to Controller\n";
  controllerBuilder->clone(*op, controllerMapping);
} // processOp FuncOp

void mock::MockQubitLocalizationPass::processOp(Builtin_UOp &uOp) {
  Operation *op = uOp.getOperation();
  llvm::outs() << "Localizing a " << op->getName() << "\n";
  int const qubitId = lookupQubitId(uOp.getTarget());

  // broadcast all classical values from Controller to all Mockss
  // recv all classical values on all Mockss
  broadcastAndReceiveValue(uOp.getTheta(), op->getLoc(), seenNodeIds);
  broadcastAndReceiveValue(uOp.getPhi(), op->getLoc(), seenNodeIds);
  broadcastAndReceiveValue(uOp.getLambda(), op->getLoc(), seenNodeIds);

  if (qubitId < 0) {
    uOp->emitOpError() << "Can't resolve qubit ID for uOp\n";
    return signalPassFailure();
  }
  // clone the gate call to the drive mock
  (*mockBuilders)[config->driveNode(qubitId)]->clone(
      *op, mockMapping[config->driveNode(qubitId)]);
} // processOp Builtin_UOp

void mock::MockQubitLocalizationPass::processOp(BuiltinCXOp &cxOp) {
  Operation *op = cxOp.getOperation();
  llvm::outs() << "Localizing a " << op->getName() << "\n";
  int const qubitId1 = lookupQubitId(cxOp.getControl());
  int const qubitId2 = lookupQubitId(cxOp.getTarget());

  if (qubitId1 < 0 || qubitId2 < 0) {
    cxOp->emitOpError() << "Can't resolve qubit ID for cxOp\n";
    return signalPassFailure();
  }
  // clone the gate call to the drive mocks
  (*mockBuilders)[config->driveNode(qubitId1)]->clone(
      *op, mockMapping[config->driveNode(qubitId1)]);
  (*mockBuilders)[config->driveNode(qubitId2)]->clone(
      *op, mockMapping[config->driveNode(qubitId2)]);
} // processOp BuiltinCXOp

void mock::MockQubitLocalizationPass::processOp(MeasureOp &measureOp) {
  Operation *op = measureOp.getOperation();
  llvm::outs() << "Localizing a " << op->getName() << "\n";
  // figure out which qubit this gate operates on
  int const qubitId = lookupQubitId(measureOp.getQubits().front());
  // clone the measure call to the drive and acquire mocks
  (*mockBuilders)[config->driveNode(qubitId)]->clone(
      *op, mockMapping[config->driveNode(qubitId)]);
  Operation *clonedOp = (*mockBuilders)[config->acquireNode(qubitId)]->clone(
      *op, mockMapping[config->acquireNode(qubitId)]);
  auto clonedMeasureOp = dyn_cast<MeasureOp>(clonedOp);

  // send the results from the acquire mock and recv on Controller
  (*mockBuilders)[config->acquireNode(qubitId)]->create<SendOp>(
      op->getLoc(), clonedMeasureOp.getOuts().front(),
      controllerBuilder->getIndexAttr(config->controllerNode()));
  auto recvOp = controllerBuilder->create<RecvOp>(
      op->getLoc(), TypeRange(clonedMeasureOp.getOuts().front().getType()),
      controllerBuilder->getIndexArrayAttr(qubitId));
  // map the result on Controller
  controllerMapping.map(measureOp.getOuts().front(), recvOp.getVals().front());
} // processOp MeasureOp

void mock::MockQubitLocalizationPass::processOp(
    CallSubroutineOp &callOp,
    std::deque<
        std::tuple<Block *, OpBuilder *,
                   std::unique_ptr<std::unordered_map<uint, OpBuilder *>>>>
        &blockAndBuilderWorkList) {
  Operation *op = callOp.getOperation();
  llvm::outs() << "Localizing a " << op->getName() << "\n";

  controllerBuilder->clone(
      *op, controllerMapping); // cloning the callOp to controller
  // first look up the func def in the parent Module
  Operation *funcOperation = SymbolTable::lookupSymbolIn(
      controllerModule->getParentOp(), callOp.getCallee());
  if (!funcOperation) {
    callOp->emitOpError() << "Unable to find func def to match "
                          << callOp.getCallee() << "\n";
    return;
  }
  auto funcOp = dyn_cast<mlir::func::FuncOp>(funcOperation);

  bool const onlyToController = classicalOnlyCheck(funcOperation);

  if (!onlyToController) {
    // broadcast all classical values from Controller to all Mockss
    // recv all classical values on all Mockss
    for (auto arg : op->getOperands()) {
      if (!arg.getType().dyn_cast<QubitType>()) {
        broadcastAndReceiveValue(arg, op->getLoc(), seenNodeIds);
      } // if not QubitType
    }   // for operands

    // Clone the subroutine call to all drive and acquire mocks
    for (uint const nodeId : seenNodeIds) {
      (*mockBuilders)[nodeId]->clone(*op, mockMapping[nodeId]);
    } // for nodeId in seenNodeIds
  }   // if !onlyToController

  // Now clone the corresponding funcOp and recurse on it
  // First check if it's already been cloned!
  if (SymbolTable::lookupSymbolIn(controllerModule.getOperation(),
                                  callOp.getCallee())) {
    llvm::outs() << callOp.getCallee() << " has already been cloned!\n";
    return;
  }
  OpBuilder::InsertPoint savedPoint = controllerBuilder->saveInsertionPoint();
  auto newBuilders = std::make_unique<std::unordered_map<uint, OpBuilder *>>();
  controllerBuilder->setInsertionPointToStart(controllerModule.getBody());
  Operation *clonedFuncOperation =
      controllerBuilder->cloneWithoutRegions(*funcOperation, controllerMapping);
  auto clonedFuncOp = dyn_cast<mlir::func::FuncOp>(clonedFuncOperation);
  if (funcOp.getCallableRegion()) {
    cloneRegionWithoutOps(&funcOp.getBody(), &clonedFuncOp.getBody(),
                          controllerMapping);
  }
  controllerBuilder->restoreInsertionPoint(savedPoint);
  if (!onlyToController) {
    for (uint const nodeId : seenNodeIds) {
      savedPoint = (*mockBuilders)[nodeId]->saveInsertionPoint();
      (*mockBuilders)[nodeId]->setInsertionPointToStart(
          dyn_cast<ModuleOp>(mockModules[nodeId]).getBody());
      Operation *clonedFuncOperation =
          (*mockBuilders)[nodeId]->cloneWithoutRegions(*funcOperation,
                                                       mockMapping[nodeId]);
      auto clonedFuncOp = dyn_cast<mlir::func::FuncOp>(clonedFuncOperation);
      if (funcOp.getCallableRegion()) {
        cloneRegionWithoutOps(&funcOp.getBody(), &clonedFuncOp.getBody(),
                              mockMapping[nodeId]);
      }
      newBuilders->emplace(nodeId, new OpBuilder(&clonedFuncOp.getBody()));
      (*mockBuilders)[nodeId]->restoreInsertionPoint(savedPoint);
    } // for nodeId in seenNodeIds
  }   // if !onlyToController
  blockAndBuilderWorkList.emplace_back(&funcOp.getBody().getBlocks().front(),
                                       new OpBuilder(&clonedFuncOp.getBody()),
                                       std::move(newBuilders));
} // processOp CallSubroutineOp

void mock::MockQubitLocalizationPass::processOp(CallGateOp &callOp) {
  Operation *op = callOp.getOperation();
  llvm::outs() << "Localizing a " << op->getName() << "\n";
  // first find out which qubits this call operates on
  // and what non-qubit values it uses
  std::vector<int> qInd;
  std::vector<Value> classicalVals;
  bool qubitIdsResolved = true;
  for (uint opInd = 0; opInd < op->getNumOperands(); ++opInd) {
    if (op->getOperand(opInd).getType().dyn_cast<QubitType>()) {
      qInd.push_back(lookupQubitId(op->getOperand(opInd)));
      if (qInd.back() < 0)
        qubitIdsResolved = false;
    }      // if operand is qubitType
    else { // non-qubit Type
      classicalVals.push_back(op->getOperand(opInd));
    } // else non-qubit Type
  }   // for operands
  if (!qubitIdsResolved) {
    callOp->emitOpError() << "Unable to resolve all qubit IDs for gate call\n";
    return signalPassFailure();
  }

  // broadcast all classical values from Controller to all Mockss
  // recv all classical values on all Mockss
  for (Value const val : classicalVals) {
    broadcastAndReceiveValue(val, op->getLoc(), seenNodeIds);
  } // for val in classicalVals

  // Clone the gate call to all relevant drive mocks
  for (uint const qubitId : qInd) {
    (*mockBuilders)[config->driveNode(qubitId)]->clone(
        *op, mockMapping[config->driveNode(qubitId)]);
  } // for qubitId in qInd
} // processOp CallGateOp

void mock::MockQubitLocalizationPass::processOp(BarrierOp &barrierOp) {
  Operation *op = barrierOp.getOperation();
  llvm::outs() << "Localizing a " << op->getName() << "\n";

  std::vector<Value> qubitOperands;
  qubitCallOperands(barrierOp, qubitOperands);

  bool qubitIdsResolved = true;
  std::vector<unsigned int> qubits;
  qubits.reserve(qubitOperands.size());
  for (auto &qubit : qubitOperands) {
    auto qubitIdx = lookupQubitId(qubit);
    qubits.emplace_back(qubitIdx);
    if (qubitIdx < 0)
      qubitIdsResolved = false;
  }

  if (!qubitIdsResolved) {
    barrierOp->emitOpError() << "Unable to resolve all qubit IDs for barrier\n";
    return signalPassFailure();
  }

  // Clone the gate call to all relevant drive
  for (uint const qubitId : qubits) {
    (*mockBuilders)[config->driveNode(qubitId)]->clone(
        *op, mockMapping[config->driveNode(qubitId)]);
  } // for qubitId in qInd

} // processOp BarrierOp

void mock::MockQubitLocalizationPass::processOp(CallDefCalGateOp &callOp) {
  Operation *op = callOp.getOperation();
  llvm::outs() << "Localizing a " << op->getName() << "\n";
  // first find out which qubits this call operates on
  // and what non-qubit values it uses
  std::vector<int> qInd;
  std::vector<Value> classicalVals;
  bool qubitIdsResolved = true;
  for (uint opInd = 0; opInd < op->getNumOperands(); ++opInd) {
    if (op->getOperand(opInd).getType().dyn_cast<QubitType>()) {
      qInd.push_back(lookupQubitId(op->getOperand(opInd)));
      if (qInd.back() < 0)
        qubitIdsResolved = false;
    }      // if operand is qubitType
    else { // non-qubit Type
      classicalVals.push_back(op->getOperand(opInd));
    } // else non-qubit Type
  }
  if (!qubitIdsResolved) {
    callOp->emitOpError() << "Unable to resolve all qubit IDs for gate call\n";
    return signalPassFailure();
  }

  // send all classical values from Controller to all Mockss
  // recv all classical values on all drive Mockss
  for (Value const val : classicalVals)
    broadcastAndReceiveValue(val, op->getLoc(), seenNodeIds);

  // Clone the gate call to all relevant drive mocks
  for (uint const qubitId : qInd) {
    (*mockBuilders)[config->driveNode(qubitId)]->clone(
        *op, mockMapping[config->driveNode(qubitId)]);
  } // for qubitId in qInd
} // processOp CallDefCalGateOp

void mock::MockQubitLocalizationPass::processOp(CallDefcalMeasureOp &callOp) {
  Operation *op = callOp.getOperation();
  llvm::outs() << "Localizing a " << op->getName() << "\n";
  // first find out which qubits this call operates on
  // and what non-qubit values it uses
  std::vector<int> qInd;
  std::vector<Value> classicalVals;
  bool qubitIdsResolved = true;
  for (uint opInd = 0; opInd < op->getNumOperands(); ++opInd) {
    if (op->getOperand(opInd).getType().dyn_cast<QubitType>()) {
      qInd.push_back(lookupQubitId(op->getOperand(opInd)));
      if (qInd.back() < 0)
        qubitIdsResolved = false;
    }      // if operand is qubitType
    else { // non-qubit Type
      classicalVals.push_back(op->getOperand(opInd));
    } // else non-qubit Type
  }
  if (!qubitIdsResolved) {
    callOp->emitOpError()
        << "Unable to resolve all qubit IDs for measurement call\n";
    return signalPassFailure();
  }

  // send all classical values from Controller to all Mockss
  // recv all classical values on all drive Mockss
  for (Value const val : classicalVals)
    broadcastAndReceiveValue(val, op->getLoc(), seenNodeIds);

  for (uint const qubitId : qInd) {
    // Clone the measure call to all drive and acquire mocks
    (*mockBuilders)[config->driveNode(qubitId)]->clone(
        *op, mockMapping[config->driveNode(qubitId)]);
    Operation *acquireOperation =
        (*mockBuilders)[config->acquireNode(qubitId)]->clone(
            *op, mockMapping[config->acquireNode(qubitId)]);
    auto acquireOp = dyn_cast<CallDefcalMeasureOp>(acquireOperation);

    // Send the measured value back to Controller and receive it on Controller
    (*mockBuilders)[config->acquireNode(qubitId)]->create<SendOp>(
        op->getLoc(), acquireOp.getRes(),
        controllerBuilder->getIndexAttr(config->controllerNode()));
    auto recvOp = controllerBuilder->create<RecvOp>(
        op->getLoc(), TypeRange(acquireOp.getRes().getType()),
        controllerBuilder->getIndexArrayAttr(qubitId));
    controllerMapping.map(callOp.getRes(), recvOp.getVals().front());
  }
} // processOp CallDefcalMeasureOp

template <class DelayOpType>
void mock::MockQubitLocalizationPass::processOp(DelayOpType &delayOp) {
  Operation *op = delayOp.getOperation();
  std::vector<int> qInd;
  bool qubitIdsResolved = true;
  for (auto operand : delayOp.getQubits()) {
    qInd.emplace_back(lookupQubitId(operand));
    if (qInd.back() < 0)
      qubitIdsResolved = false;
  }
  if (!qubitIdsResolved) {
    delayOp->emitOpError() << "Unable to resolve all qubit IDs for delay\n";
    return signalPassFailure();
  }
  if (delayOp.getQubits().empty()) // no qubit args means all qubits
    for (uint const qId : seenQubitIds)
      qInd.emplace_back((int)qId);

  // turn the vector of qubitIds into a set of node Ids
  std::unordered_set<uint> involvedNodes;
  for (int const qubitId : qInd) {
    involvedNodes.emplace(config->driveNode(qubitId));
    involvedNodes.emplace(config->acquireNode(qubitId));
  }

  if (auto dOp = dyn_cast<DelayOp>(op)) {
    auto *durationDeclare = dOp.getTime().getDefiningOp();
    for (uint const id : involvedNodes)
      (*mockBuilders)[id]->clone(*durationDeclare, mockMapping[id]);
  }

  if (delayOp.getQubits().empty())
    controllerBuilder->clone(*op, controllerMapping);
  // clone the delay op to the involved nodes
  for (uint const nodeId : involvedNodes)
    (*mockBuilders)[nodeId]->clone(*op, mockMapping[nodeId]);
} // processOp DelayOp

void mock::MockQubitLocalizationPass::processOp(
    mlir::func::ReturnOp &returnOp) {
  Operation *op = returnOp.getOperation();
  controllerBuilder->clone(*op, controllerMapping);
  FlatSymbolRefAttr const symbolRef = SymbolRefAttr::get(op->getParentOp());
  if (symbolRef && symbolRef.getLeafReference() == "main")
    for (auto arg : returnOp.getOperands())
      broadcastAndReceiveValue(arg, op->getLoc(), seenNodeIds);
  if (!classicalOnlyCheck(op->getParentOp()))
    for (uint const nodeId : seenNodeIds)
      (*mockBuilders)[nodeId]->clone(*op, mockMapping[nodeId]);
} // processOp ReturnOp

void mock::MockQubitLocalizationPass::processOp(scf::YieldOp &yieldOp) {
  Operation *op = yieldOp.getOperation();
  for (auto arg : yieldOp.getResults())
    broadcastAndReceiveValue(arg, op->getLoc(), seenNodeIds);
  // copy op to all nodes
  controllerBuilder->clone(*op, controllerMapping);
  if (!classicalOnlyCheck(op->getParentOp()))
    for (auto nodeId : seenNodeIds)
      (*mockBuilders)[nodeId]->clone(*op, mockMapping[nodeId]);
} // processOp YieldOp

void mock::MockQubitLocalizationPass::processOp(
    scf::IfOp &ifOp,
    std::deque<
        std::tuple<Block *, OpBuilder *,
                   std::unique_ptr<std::unordered_map<uint, OpBuilder *>>>>
        &blockAndBuilderWorkList) {
  Operation *op = ifOp.getOperation();
  llvm::outs() << "Localizing an " << op->getName()
               << " operation, recursing into subregions\n";

  if (classicalOnlyCheck(op)) {
    // only classical ops, everything goes on controller
    controllerBuilder->clone(*op, controllerMapping);
    return;
  }

  llvm::outs() << "Found a quantum ifOp!\n";
  // first broadcast the condition value from Controller to Mockss
  // then clone the if op everywhere but with empty blocks
  auto newThenBuilders =
      std::make_unique<std::unordered_map<uint, OpBuilder *>>();
  auto newElseBuilders =
      std::make_unique<std::unordered_map<uint, OpBuilder *>>();

  // check if the condition is the result of a single measurement
  auto measureOp = ifOp.getCondition().getDefiningOp<MeasureOp>();
  int savedQubitId = -1;
  if (measureOp) { // remove the drive node from seenNodeIds temporarily
    // only if it can be resolved
    savedQubitId = lookupQubitId(measureOp.getQubits().front());
    if (savedQubitId >= 0) {
      seenNodeIds.erase(config->driveNode(savedQubitId));

      // receive the measurement result directly from the acquireNode
      auto recvOp =
          (*mockBuilders)[config->driveNode(savedQubitId)]->create<RecvOp>(
              measureOp->getLoc(), TypeRange(ifOp.getCondition().getType()),
              controllerBuilder->getIndexArrayAttr(
                  config->acquireNode(savedQubitId)));
      // map the result on the drive node
      mockMapping[config->driveNode(savedQubitId)].map(
          ifOp.getCondition(), recvOp.getVals().front());
    }
  }

  broadcastAndReceiveValue(ifOp.getCondition(), op->getLoc(), seenNodeIds);

  // now restore the erased drive node id
  if (measureOp && savedQubitId != -1)
    seenNodeIds.emplace(config->driveNode(savedQubitId));

  Operation *clonedOp =
      controllerBuilder->cloneWithoutRegions(*op, controllerMapping);
  auto clonedIfOp = dyn_cast<scf::IfOp>(clonedOp);
  if (!ifOp.getThenRegion().empty()) {
    cloneRegionWithoutOps(&ifOp.getThenRegion(), &clonedIfOp.getThenRegion(),
                          controllerMapping);
  }
  if (!ifOp.getElseRegion().empty()) {
    cloneRegionWithoutOps(&ifOp.getElseRegion(), &clonedIfOp.getElseRegion(),
                          controllerMapping);
  }
  for (uint const nodeId : seenNodeIds) {
    Operation *clonedOp =
        (*mockBuilders)[nodeId]->cloneWithoutRegions(*op, mockMapping[nodeId]);
    auto clonedIfOp = dyn_cast<scf::IfOp>(clonedOp);
    if (!ifOp.getThenRegion().empty()) {
      cloneRegionWithoutOps(&ifOp.getThenRegion(), &clonedIfOp.getThenRegion(),
                            mockMapping[nodeId]);
      newThenBuilders->emplace(nodeId,
                               new OpBuilder(clonedIfOp.getThenRegion()));
    }
    if (!ifOp.getElseRegion().empty()) {
      cloneRegionWithoutOps(&ifOp.getElseRegion(), &clonedIfOp.getElseRegion(),
                            mockMapping[nodeId]);
      newElseBuilders->emplace(nodeId,
                               new OpBuilder(clonedIfOp.getElseRegion()));
    }
  } // for nodeId : seenNodeIds
  if (!ifOp.getThenRegion().empty()) {
    llvm::outs() << "Pushing onto blockAndBuilderWorkList! Then region\n";
    blockAndBuilderWorkList.emplace_back(
        &ifOp.getThenRegion().getBlocks().front(),
        new OpBuilder(clonedIfOp.getThenRegion()), std::move(newThenBuilders));
  }
  if (!ifOp.getElseRegion().empty()) {
    llvm::outs() << "Pushing onto blockAndBuilderWorkList! Else region\n";
    blockAndBuilderWorkList.emplace_back(
        &ifOp.getElseRegion().getBlocks().front(),
        new OpBuilder(clonedIfOp.getElseRegion()), std::move(newElseBuilders));
  }
} // processOp scf::IfOp

void mock::MockQubitLocalizationPass::processOp(
    scf::ForOp &forOp,
    std::deque<
        std::tuple<Block *, OpBuilder *,
                   std::unique_ptr<std::unordered_map<uint, OpBuilder *>>>>
        &blockAndBuilderWorkList) {
  Operation *op = forOp.getOperation();

  llvm::outs() << "Localizing an " << op->getName()
               << " operation, recursing into subregions\n";
  if (classicalOnlyCheck(op)) {
    // only classical ops, everything goes on controller
    controllerBuilder->clone(*op, controllerMapping);
  } else { // else some quantum ops
    llvm::outs() << "Found a quantum forOp!\n";
    // first broadcast the lb, ub, step, and init args
    // from Controller to Mockss then clone the for op everywhere
    // but with empty blocks
    auto newBuilders =
        std::make_unique<std::unordered_map<uint, OpBuilder *>>();
    broadcastAndReceiveValue(forOp.getLowerBound(), op->getLoc(), seenNodeIds);
    broadcastAndReceiveValue(forOp.getUpperBound(), op->getLoc(), seenNodeIds);
    broadcastAndReceiveValue(forOp.getStep(), op->getLoc(), seenNodeIds);
    for (auto arg : forOp.getInitArgs())
      broadcastAndReceiveValue(arg, op->getLoc(), seenNodeIds);

    Operation *clonedOp =
        controllerBuilder->cloneWithoutRegions(*op, controllerMapping);
    auto clonedForOp = dyn_cast<scf::ForOp>(clonedOp);
    cloneRegionWithoutOps(&forOp.getLoopBody(), &clonedForOp.getLoopBody(),
                          controllerMapping);
    for (uint const nodeId : seenNodeIds) {
      Operation *clonedOp = (*mockBuilders)[nodeId]->cloneWithoutRegions(
          *op, mockMapping[nodeId]);
      auto clonedFor = dyn_cast<scf::ForOp>(clonedOp);
      cloneRegionWithoutOps(&forOp.getLoopBody(), &clonedFor.getLoopBody(),
                            mockMapping[nodeId]);
      newBuilders->emplace(nodeId, new OpBuilder(clonedFor.getLoopBody()));
    } // for nodeId : seenNodeIds
    blockAndBuilderWorkList.emplace_back(
        &forOp.getLoopBody().getBlocks().front(),
        new OpBuilder(clonedForOp.getLoopBody()), std::move(newBuilders));
  } // else some quantum ops
} // processOp scf::ForOp

// Entry point for the pass.
void mock::MockQubitLocalizationPass::runOnOperation(MockSystem &target) {
  // This pass is only called on the top-level module Op
  Operation *moduleOp = getOperation();
  config = &target.getConfig();

  ModuleOp topModuleOp = dyn_cast<ModuleOp>(moduleOp);
  Operation *mainFunc = getMainFunction(moduleOp);
  if (!mainFunc) {
    moduleOp->emitOpError()
        << "Error: No main function found, cannot localize qubits\n";
    return;
  }
  if (!config) {
    topModuleOp->emitError() << "No config available\n";
    return signalPassFailure();
  }

  // Initialize the Controller Module
  auto b = OpBuilder::atBlockEnd(topModuleOp.getBody());
  // ModuleOp test = ModuleOp::create(b.getUnknownLoc());
  controllerModule = dyn_cast<ModuleOp>(
      b.create<ModuleOp>(b.getUnknownLoc(), llvm::StringRef("controller"))
          .getOperation());
  mlir::func::FuncOp controllerMainOp =
      addMainFunction(controllerModule.getOperation(), mainFunc->getLoc());
  controllerBuilder = new OpBuilder(controllerMainOp.getBody());
  controllerModule->setAttr(
      llvm::StringRef("quir.nodeId"),
      controllerBuilder->getUI32IntegerAttr(config->controllerNode()));
  controllerModule->setAttr(
      llvm::StringRef("quir.nodeType"),
      controllerBuilder->getStringAttr(llvm::StringRef("controller")));

  // first work on creating the modules
  // We do this first so that we can detect all physical qubit declarations
  auto newBuilders = std::make_unique<std::unordered_map<uint, OpBuilder *>>();

  for (const auto &result : llvm::enumerate(config->getDriveNodes())) {
    uint const qubitIdx = result.index();
    uint const nodeId = result.value();
    llvm::outs() << "Creating module for drive Mocks " << qubitIdx << "\n";
    auto driveMod = b.create<ModuleOp>(
        b.getUnknownLoc(),
        llvm::StringRef("mock_drive_" + std::to_string(qubitIdx)));
    driveMod.getOperation()->setAttr(
        llvm::StringRef("quir.nodeType"),
        controllerBuilder->getStringAttr(llvm::StringRef("drive")));
    driveMod.getOperation()->setAttr(
        llvm::StringRef("quir.nodeId"),
        controllerBuilder->getUI32IntegerAttr(nodeId));
    driveMod.getOperation()->setAttr(
        llvm::StringRef("quir.physicalId"),
        controllerBuilder->getI32IntegerAttr(qubitIdx));
    mockModules[nodeId] = driveMod.getOperation();
    mlir::func::FuncOp mockMainOp =
        addMainFunction(driveMod.getOperation(), mainFunc->getLoc());
    newBuilders->emplace(nodeId, new OpBuilder(mockMainOp.getBody()));
  }

  for (const auto &result : llvm::enumerate(config->getAcquireNodes())) {
    uint const acquireIdx = result.index();
    uint const nodeId = result.value();
    llvm::outs() << "Creating module for acquire Mocks " << acquireIdx << "\n";
    auto acquireMod = b.create<ModuleOp>(
        b.getUnknownLoc(),
        llvm::StringRef("mock_acquire_" + std::to_string(acquireIdx)));
    acquireMod.getOperation()->setAttr(
        llvm::StringRef("quir.nodeType"),
        controllerBuilder->getStringAttr(llvm::StringRef("acquire")));
    acquireMod.getOperation()->setAttr(
        llvm::StringRef("quir.nodeId"),
        controllerBuilder->getUI32IntegerAttr(nodeId));
    acquireMod.getOperation()->setAttr(
        llvm::StringRef("quir.physicalIds"),
        controllerBuilder->getI32ArrayAttr(
            ArrayRef<int>(config->acquireQubits(nodeId))));
    mockModules[nodeId] = acquireMod.getOperation();
    mlir::func::FuncOp mockMainOp =
        addMainFunction(acquireMod.getOperation(), mainFunc->getLoc());
    newBuilders->emplace(nodeId, new OpBuilder(mockMainOp.getBody()));
  }

  mainFunc->walk([&](DeclareQubitOp qubitOp) {
    llvm::outs() << qubitOp.getOperation()->getName()
                 << " id: " << qubitOp.getId() << "\n";
    if (!qubitOp.getId().has_value() ||
        qubitOp.getId().value() > config->getNumQubits()) {
      qubitOp->emitOpError()
          << "Error! Found a qubit without an ID or with ID > "
          << std::to_string(config->getNumQubits())
          << " (the number of qubits in the config)"
          << " during qubit localization!\n";
      signalPassFailure();
    }
    uint const qId = qubitOp.getId().value();
    seenQubitIds.emplace(qId);
    driveNodeIds.emplace(config->driveNode(qId));
    acquireNodeIds.emplace(config->acquireNode(qId));
    seenNodeIds.emplace(config->driveNode(qId));
    seenNodeIds.emplace(config->acquireNode(qId));
  });

  // refill the worklist
  std::deque<std::tuple<Block *, OpBuilder *,
                        std::unique_ptr<std::unordered_map<uint, OpBuilder *>>>>
      blockAndBuilderWorkList;
  for (Region &region : mainFunc->getRegions()) {
    for (Block &block : region.getBlocks()) {
      blockAndBuilderWorkList.emplace_back(&block, controllerBuilder,
                                           std::move(newBuilders));
    }
  }

  while (!blockAndBuilderWorkList.empty()) {
    llvm::outs() << "Entering blockAndBuilderWorklist body!\n";
    Block *block = std::get<0>(blockAndBuilderWorkList.front());
    controllerBuilder = std::get<1>(blockAndBuilderWorkList.front());
    mockBuilders = std::get<2>(blockAndBuilderWorkList.front()).release();
    blockAndBuilderWorkList.pop_front();
    for (Operation &op : block->getOperations()) {
      if (auto qubitOp = dyn_cast<DeclareQubitOp>(op)) {
        processOp(qubitOp);
      } else if (auto resetOp = dyn_cast<ResetQubitOp>(op)) {
        processOp(resetOp);
      } else if (auto uOp = dyn_cast<Builtin_UOp>(op)) {
        processOp(uOp);
      } else if (auto cxOp = dyn_cast<BuiltinCXOp>(op)) {
        processOp(cxOp);
      } else if (auto measureOp = dyn_cast<MeasureOp>(op)) {
        processOp(measureOp);
      } else if (auto callOp = dyn_cast<CallSubroutineOp>(op)) {
        processOp(callOp, blockAndBuilderWorkList);
      } else if (auto callOp = dyn_cast<CallGateOp>(op)) {
        processOp(callOp);
      } else if (auto callOp = dyn_cast<BarrierOp>(op)) {
        processOp(callOp);
      } else if (auto callOp = dyn_cast<CallDefCalGateOp>(op)) {
        processOp(callOp);
      } else if (auto callOp = dyn_cast<CallDefcalMeasureOp>(op)) {
        processOp(callOp);
      } else if (auto delayOp = dyn_cast<DelayOp>(op)) {
        processOp(delayOp);
      } else if (auto delayOp = dyn_cast<DelayCyclesOp>(op)) {
        processOp(delayOp);
      } else if (auto returnOp = dyn_cast<mlir::func::ReturnOp>(op)) {
        processOp(returnOp);
      } else if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
        processOp(yieldOp);
      } else if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
        processOp(ifOp, blockAndBuilderWorkList);
      } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
        processOp(forOp, blockAndBuilderWorkList);
      } else if (dyn_cast<mlir::func::FuncOp>(op) || dyn_cast<ModuleOp>(op)) {
        // do nothing
      }      // moduleOp
      else { // some classical op, should go to Controller
        auto *clonedOp = controllerBuilder->clone(op, controllerMapping);
        // now add mappping for all results from the original op to the
        // clonedOp
        for (uint index = 0; index < op.getNumResults(); ++index) {
          controllerMapping.map(op.getResult(index),
                                clonedOp->getResult(index));
        }
      } // some classical op
    }   // for Operations
    // delete the allocated opbuilders
    delete controllerBuilder;
    for (uint const nodeId : seenNodeIds)
      delete (*mockBuilders)[nodeId];
    delete mockBuilders;
  } // while !blockAndBuilderWorklist.empty()

  cloneVariableDeclarations(topModuleOp);
} // runOnOperation()

void mock::MockQubitLocalizationPass::cloneVariableDeclarations(
    mlir::ModuleOp topModuleOp) {

  mlir::OpBuilder controllerModuleBuilder(controllerModule.getBodyRegion());

  // clone variable declarations into all target modules
  for (auto variableDeclaration : topModuleOp.getOps<DeclareVariableOp>())
    controllerModuleBuilder.clone(*variableDeclaration.getOperation());
}

llvm::StringRef MockQubitLocalizationPass::getArgument() const {
  return "mock-qubit-localization";
}

llvm::StringRef MockQubitLocalizationPass::getDescription() const {
  return "Create modules for Mock code blocks.";
}

llvm::StringRef MockQubitLocalizationPass::getName() const {
  return "Mock Qubit Localization Pass";
}
