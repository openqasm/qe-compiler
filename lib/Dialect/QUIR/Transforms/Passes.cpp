//===- Passes.cpp - QUIR Passes ---------------------------------*- C++ -*-===//
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

#include "Dialect/QUIR/Transforms/Passes.h"
#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTestInterfaces.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"
#include "Dialect/QUIR/Transforms/AddShotLoop.h"
#include "Dialect/QUIR/Transforms/AngleConversion.h"
#include "Dialect/QUIR/Transforms/BreakReset.h"
#include "Dialect/QUIR/Transforms/ConvertDurationUnits.h"
#include "Dialect/QUIR/Transforms/FunctionArgumentSpecialization.h"
#include "Dialect/QUIR/Transforms/LoadElimination.h"
#include "Dialect/QUIR/Transforms/MergeCircuitMeasures.h"
#include "Dialect/QUIR/Transforms/MergeCircuits.h"
#include "Dialect/QUIR/Transforms/MergeMeasures.h"
#include "Dialect/QUIR/Transforms/MergeParallelResets.h"
#include "Dialect/QUIR/Transforms/QuantumDecoration.h"
#include "Dialect/QUIR/Transforms/RemoveQubitOperands.h"
#include "Dialect/QUIR/Transforms/RemoveUnusedCircuits.h"
#include "Dialect/QUIR/Transforms/ReorderCircuits.h"
#include "Dialect/QUIR/Transforms/ReorderMeasurements.h"
#include "Dialect/QUIR/Transforms/SubroutineCloning.h"
#include "Dialect/QUIR/Transforms/UnusedVariable.h"
#include "Dialect/QUIR/Transforms/VariableElimination.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>

using namespace mlir;
using namespace mlir::quir;
using namespace mlir::oq3;

namespace mlir {
struct InlinerPass;
} // namespace mlir

namespace mlir::quir {
/// This pass illustrates the IR nesting through printing.
struct TestPrintNestingPass
    : public PassWrapper<TestPrintNestingPass, OperationPass<>> {
  // Entry point for the pass.
  void runOnOperation() override {
    Operation *op = getOperation();
    resetIndent();
    printOperation(op);
  }

  /// The three methods below are mutually recursive and follow the nesting of
  /// the IR: operation->region->block->operation->...
  // NOLINTNEXTLINE(misc-no-recursion)
  void printOperation(Operation *op) {
    // Print the operation itself and some of its properties
    printIndent() << "visiting op: '" << op->getName() << "' with "
                  << op->getNumOperands() << " operands and "
                  << op->getNumResults() << " results\n";
    // Print the operation attributes
    if (!op->getAttrs().empty()) {
      printIndent() << op->getAttrs().size() << " attributes:\n";
      for (NamedAttribute const attr : op->getAttrs())
        printIndent() << " - '" << attr.getName() << "' : '" << attr.getValue()
                      << "'\n";
    }

    // Recurse into each of the regions attached to the operation.
    printIndent() << " " << op->getNumRegions() << " nested regions:\n";
    auto indent = pushIndent();
    for (Region &region : op->getRegions())
      printRegion(region);
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  void printRegion(Region &region) {
    // A region does not hold anything by itself other than a list of blocks.
    printIndent() << "Region with " << region.getBlocks().size()
                  << " blocks:\n";
    auto indent = pushIndent();
    for (Block &block : region.getBlocks())
      printBlock(block);
  }

  // NOLINTNEXTLINE(misc-no-recursion)
  void printBlock(Block &block) {
    // Print the block intrinsics properties (basically: argument list)
    printIndent()
        << "Block with " << block.getNumArguments() << " arguments, "
        << block.getNumSuccessors()
        << " successors, and "
        // Note, this `.size()` is traversing a linked-list and is O(n).
        << block.getOperations().size() << " operations\n";

    // Block main role is to hold a list of Operations: let's recurse.
    auto indent = pushIndent();
    for (Operation &op : block.getOperations())
      printOperation(&op);
  }

  /// Manages the indentation as we traverse the IR nesting.
  int indent;
  struct IdentRAII {
    int &indent;
    IdentRAII(int &indent) : indent(indent) {}
    ~IdentRAII() { --indent; }
  };
  void resetIndent() { indent = 0; }
  auto pushIndent() -> IdentRAII { return {++indent}; }

  auto printIndent() -> llvm::raw_ostream & {
    for (int i = 0; i < indent; ++i)
      llvm::outs() << "  ";
    return llvm::outs();
  }

  llvm::StringRef getArgument() const override { return "test-print-nesting"; }
  llvm::StringRef getDescription() const override {
    return "Test various printing.";
  }
}; // end struct TestPrintNestingPass

///////////////// ClassicalOnlyDetectionPass functions /////////////////////
// detects whether or not an operation contains quantum operations inside
auto ClassicalOnlyDetectionPass::hasQuantumSubOps(Operation *inOp) -> bool {
  bool retVal = true;
  inOp->walk([&](Operation *op) {
    if (dyn_cast<BuiltinCXOp>(op) || dyn_cast<Builtin_UOp>(op) ||
        dyn_cast<CallDefCalGateOp>(op) || dyn_cast<CallDefcalMeasureOp>(op) ||
        dyn_cast<DelayOp>(op) || dyn_cast<CallGateOp>(op) ||
        dyn_cast<MeasureOp>(op) || dyn_cast<DeclareQubitOp>(op) ||
        dyn_cast<ResetQubitOp>(op) || dyn_cast<CallCircuitOp>(op)) {
      retVal = false;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return retVal;
} // ClassicalOnlyDetectionPass::hasQuantumSubOps

// Entry point for the ClassicalOnlyDetectionPass pass
void ClassicalOnlyDetectionPass::runOnOperation() {
  // This pass is only called on the top-level module Op
  Operation *moduleOperation = getOperation();
  OpBuilder b(moduleOperation);

  moduleOperation->walk([&](Operation *op) {
    if (dyn_cast<scf::IfOp>(op) || dyn_cast<scf::ForOp>(op) ||
        dyn_cast<scf::WhileOp>(op) || dyn_cast<quir::SwitchOp>(op) ||
        dyn_cast<quir::CircuitOp>(op))
      op->setAttr(llvm::StringRef("quir.classicalOnly"),
                  b.getBoolAttr(hasQuantumSubOps(op)));
    if (auto funcOp = dyn_cast<mlir::func::FuncOp>(op)) {
      // just check the arguments for qubitType values
      FunctionType const fType = funcOp.getFunctionType();
      bool const isMain =
          SymbolRefAttr::get(funcOp).getLeafReference() == "main";
      bool quantumOperands = false;
      for (auto argType : fType.getInputs()) {
        if (argType.isa<QubitType>()) {
          quantumOperands = true;
          break;
        }
      }
      bool quantumDeclarations = false;
      funcOp->walk([&](DeclareQubitOp op) { quantumDeclarations = true; });
      funcOp->walk([&](CallCircuitOp op) { quantumDeclarations = true; });
      op->setAttr(
          llvm::StringRef("quir.classicalOnly"),
          b.getBoolAttr(!quantumOperands && !quantumDeclarations && !isMain));
    } // if funcOp
  });
} // ClassicalOnlyDetectionPass::runOnOperation

llvm::StringRef ClassicalOnlyDetectionPass::getArgument() const {
  return "classical-only-detection";
}
llvm::StringRef ClassicalOnlyDetectionPass::getDescription() const {
  return "Detect control flow blocks that contain only classical (non-quantum) "
         "operations, and decorate them with a classicalOnly bool attribute";
}

llvm::StringRef ClassicalOnlyDetectionPass::getName() const {
  return "Classical Only Detection Pass";
}

/////////////// End ClassicalOnlyDetectionPass functions ///////////////////

struct DumpVariableDominanceInfoPass
    : public PassWrapper<DumpVariableDominanceInfoPass, OperationPass<>> {

  void runOnOperation() override {
    Operation *op = getOperation();
    auto &domInfo = getAnalysis<mlir::DominanceInfo>();

    op->walk([&](mlir::oq3::DeclareVariableOp decl) {
      auto *symbolTable = mlir::SymbolTable::getNearestSymbolTable(decl);
      auto symbolUses = mlir::SymbolTable::getSymbolUses(decl, symbolTable);
      SmallVector<Operation *, 4> varAssignments;
      SmallVector<Operation *, 4> varUses;

      decl.dump();

      // collect variable assignments and variable uses
      for (auto use : *symbolUses) {
        Operation *userOp = use.getUser();

        if (mlir::isa<mlir::oq3::VariableAssignOp>(userOp) ||
            mlir::isa<mlir::oq3::CBitAssignBitOp>(userOp))
          varAssignments.push_back(use.getUser());
        else if (mlir::isa<mlir::oq3::VariableLoadOp>(userOp) ||
                 mlir::isa<mlir::oq3::CBitExtractBitOp>(userOp))
          varUses.push_back(userOp);
      }

      for (Operation *assignment : varAssignments) {
        llvm::errs() << "  ";
        assignment->dump();

        for (Operation *use : varUses) {
          bool const dominates = domInfo.dominates(assignment, use);
          llvm::errs() << "    "
                       << (dominates ? "dominates" : "no dominance over")
                       << " ";
          use->dump();
        }
      }
    });
  }

  llvm::StringRef getArgument() const override {
    return "quir-dump-var-dominfo";
  }
  llvm::StringRef getDescription() const override {
    return "Dump dominance info for QUIR variable operations.";
  }
};

void quirPassPipelineBuilder(OpPassManager &pm) {
  pm.addPass(std::make_unique<LoadEliminationPass>());
  pm.addPass(std::make_unique<ClassicalOnlyDetectionPass>());

  // TODO: Decide if we want to enable the inliner pass in this pipeline
  // pm.addPass(mlir::createInlinerPass());
}

void registerQuirPasses() {
  //===----------------------------------------------------------------------===//
  // Transform Passes
  //===----------------------------------------------------------------------===//
  PassRegistration<quir::TestPrintNestingPass>();
  PassRegistration<quir::FunctionArgumentSpecializationPass>();
  PassRegistration<quir::ClassicalOnlyDetectionPass>();
  PassRegistration<quir::BreakResetPass>();
  PassRegistration<quir::MergeResetsLexicographicPass>();
  PassRegistration<quir::MergeResetsTopologicalPass>();
  PassRegistration<quir::SubroutineCloningPass>();
  PassRegistration<quir::RemoveQubitOperandsPass>();
  PassRegistration<quir::RemoveUnusedCircuitsPass>();
  PassRegistration<quir::UnusedVariablePass>();
  PassRegistration<quir::AddShotLoopPass>();
  PassRegistration<quir::QuantumDecorationPass>();
  PassRegistration<quir::ReorderMeasurementsPass>();
  PassRegistration<quir::ReorderCircuitsPass>();
  PassRegistration<quir::MergeCircuitsPass>();
  PassRegistration<quir::MergeCircuitMeasuresTopologicalPass>();
  PassRegistration<quir::MergeMeasuresLexographicalPass>();
  PassRegistration<quir::MergeMeasuresTopologicalPass>();
  PassRegistration<quir::QUIRAngleConversionPass>();
  PassRegistration<quir::LoadEliminationPass>();
  PassRegistration<quir::DumpVariableDominanceInfoPass>();
  PassRegistration<quir::VariableEliminationPass>();
  PassRegistration<quir::ConvertDurationUnitsPass>();

  //===----------------------------------------------------------------------===//
  // Test Passes
  //===----------------------------------------------------------------------===//
  PassRegistration<quir::TestQubitOpInterfacePass>();
}

void registerQuirPassPipeline() {
  PassPipelineRegistration<> const pipeline(
      "quirOpt", "Enable QUIR-specific optimizations",
      quir::quirPassPipelineBuilder);
}
} // end namespace mlir::quir
