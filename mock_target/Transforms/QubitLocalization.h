//===- QubitLocalization.h - Modules for qubit control ----------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2023.
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

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"

#include <deque>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace qssc::targets::mock {

struct MockQubitLocalizationPass
    : public mlir::PassWrapper<MockQubitLocalizationPass,
                               qssc::hal::TargetOperationPass<MockSystem>> {

  void processOp(mlir::quir::DeclareQubitOp &qubitOp);
  void processOp(mlir::quir::ResetQubitOp &resetOp);
  void processOp(mlir::FuncOp &funcOp);
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
  void processOp(mlir::ReturnOp &returnOp);
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
                             mlir::BlockAndValueMapping &mapper);
  void cloneRegionWithoutOps(mlir::Region *from, mlir::Region *dest,
                             mlir::Region::iterator destPos,
                             mlir::BlockAndValueMapping &mapper);
  auto addMainFunction(mlir::Operation *moduleOperation,
                       const mlir::Location &loc) -> mlir::FuncOp;
  void cloneVariableDeclarations(mlir::ModuleOp topModuleOp);

  MockConfig *config;
  mlir::ModuleOp controllerModule;
  mlir::BlockAndValueMapping controllerMapping;
  mlir::OpBuilder *controllerBuilder;
  std::unordered_set<uint> seenNodeIds;
  std::unordered_set<uint> seenQubitIds;
  std::unordered_set<uint> acquireNodeIds;
  std::unordered_set<uint> driveNodeIds;

  mlir::DenseSet<mlir::Value> alreadyBroadcastValues;
  std::unordered_map<uint, mlir::Operation *> mockModules;   // one per nodeId
  std::unordered_map<uint, mlir::OpBuilder *> *mockBuilders; // one per nodeId
  std::unordered_map<uint, mlir::BlockAndValueMapping>
      mockMapping; // one per nodeId

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
}; // struct MockQubitLocalizationPass

} // namespace qssc::targets::mock

#endif // MOCK_QUBIT_LOCALIZATION_H
