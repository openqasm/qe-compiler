//===- QubitLocalization.h - Modules for qubit control ----------*- C++ -*-===//
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
//  This file declares the pass for creating localized qubit modules
//
//===----------------------------------------------------------------------===//

#ifndef MOCK_QUBIT_LOCALIZATION_H
#define MOCK_QUBIT_LOCALIZATION_H

#include "MockTarget.h"

#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "HAL/TargetOperationPass.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace qssc::targets::systems::mock {

struct MockQubitLocalizationPass
    : public mlir::PassWrapper<MockQubitLocalizationPass,
                               qssc::hal::TargetOperationPass<MockSystem>> {

  void processOp(mlir::quir::DeclareQubitOp &qubitOp);
  void processOp(mlir::quir::ResetQubitOp &resetOp);
  void processOp(mlir::func::FuncOp &funcOp);
  void processOp(mlir::quir::Builtin_UOp &uOp);
  void processOp(mlir::quir::BuiltinCXOp &cxOp);
  void processOp(mlir::quir::MeasureOp &measureOp);
  void
  processOp(mlir::quir::CallSubroutineOp &callOp,
            std::deque<std::tuple<
                mlir::Block *, mlir::OpBuilder *,
                std::unique_ptr<std::unordered_map<uint, mlir::OpBuilder *>>>>
                &blockAndBuilderWorkList);
  void processOp(mlir::quir::CallGateOp &callOp);
  void processOp(mlir::quir::BarrierOp &callOp);
  void processOp(mlir::quir::CallDefCalGateOp &callOp);
  void processOp(mlir::quir::CallDefcalMeasureOp &callOp);
  template <class DelayOpType>
  void processOp(DelayOpType &delayOp);
  void processOp(mlir::func::ReturnOp &returnOp);
  void processOp(mlir::scf::YieldOp &yieldOp);
  void
  processOp(mlir::scf::IfOp &ifOp,
            std::deque<std::tuple<
                mlir::Block *, mlir::OpBuilder *,
                std::unique_ptr<std::unordered_map<uint, mlir::OpBuilder *>>>>
                &blockAndBuilderWorkList);
  void
  processOp(mlir::scf::ForOp &forOp,
            std::deque<std::tuple<
                mlir::Block *, mlir::OpBuilder *,
                std::unique_ptr<std::unordered_map<uint, mlir::OpBuilder *>>>>
                &blockAndBuilderWorkList);

  void runOnOperation(MockSystem &target) override;
  auto lookupQubitId(const mlir::Value &val) -> int;
  void broadcastAndReceiveValue(const mlir::Value &val,
                                const mlir::Location &loc,
                                const std::unordered_set<uint> &toNodeIds);
  void cloneRegionWithoutOps(mlir::Region *from, mlir::Region *dest,
                             mlir::IRMapping &mapper);
  void cloneRegionWithoutOps(mlir::Region *from, mlir::Region *dest,
                             mlir::Region::iterator destPos,
                             mlir::IRMapping &mapper);
  auto addMainFunction(mlir::Operation *moduleOperation,
                       const mlir::Location &loc) -> mlir::func::FuncOp;
  void cloneVariableDeclarations(mlir::ModuleOp topModuleOp);

  MockConfig *config;
  mlir::ModuleOp controllerModule;
  mlir::IRMapping controllerMapping;
  mlir::OpBuilder *controllerBuilder;
  std::unordered_set<uint> seenNodeIds;
  std::unordered_set<uint> seenQubitIds;
  std::unordered_set<uint> acquireNodeIds;
  std::unordered_set<uint> driveNodeIds;

  mlir::DenseSet<mlir::Value> alreadyBroadcastValues;
  std::unordered_map<uint, mlir::Operation *> mockModules;   // one per nodeId
  std::unordered_map<uint, mlir::OpBuilder *> *mockBuilders; // one per nodeId
  std::unordered_map<uint, mlir::IRMapping> mockMapping;     // one per nodeId

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
}; // struct MockQubitLocalizationPass

} // namespace qssc::targets::systems::mock

#endif // MOCK_QUBIT_LOCALIZATION_H
