//===- Passes.cpp - QUIR Passes ---------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <list>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"
#include "Dialect/QUIR/Transforms/Passes.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::quir;

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
      for (NamedAttribute attr : op->getAttrs())
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
        dyn_cast<ResetQubitOp>(op)) {
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
        dyn_cast<quir::SwitchOp>(op))
      op->setAttr(llvm::StringRef("quir.classicalOnly"),
                  b.getBoolAttr(hasQuantumSubOps(op)));
    if (auto funcOp = dyn_cast<FuncOp>(op)) {
      // just check the arguments for qubitType values
      FunctionType fType = funcOp.getType();
      bool isMain = SymbolRefAttr::get(funcOp).getLeafReference() == "main";
      bool quantumArgs = false;
      for (auto argType : fType.getInputs()) {
        if (argType.isa<QubitType>()) {
          quantumArgs = true;
          break;
        }
      }
      bool quantumDeclarations = false;
      funcOp->walk([&](DeclareQubitOp op) { quantumDeclarations = true; });
      op->setAttr(
          llvm::StringRef("quir.classicalOnly"),
          b.getBoolAttr(!quantumArgs && !quantumDeclarations && !isMain));
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
/////////////// End ClassicalOnlyDetectionPass functions ///////////////////

struct DumpVariableDominanceInfoPass
    : public PassWrapper<DumpVariableDominanceInfoPass, OperationPass<>> {

  void runOnOperation() override {
    Operation *op = getOperation();
    auto &domInfo = getAnalysis<mlir::DominanceInfo>();

    op->walk([&](mlir::quir::DeclareVariableOp decl) {
      auto *symbolTable = mlir::SymbolTable::getNearestSymbolTable(decl);
      auto symbolUses = mlir::SymbolTable::getSymbolUses(decl, symbolTable);
      SmallVector<Operation *, 4> varAssignments;
      SmallVector<Operation *, 4> varUses;

      decl.dump();

      // collect variable assignments and variable uses
      for (auto use : *symbolUses) {
        Operation *userOp = use.getUser();

        if (mlir::isa<mlir::quir::VariableAssignOp>(userOp) ||
            mlir::isa<mlir::quir::AssignCbitBitOp>(userOp))
          varAssignments.push_back(use.getUser());
        else if (mlir::isa<mlir::quir::UseVariableOp>(userOp) ||
                 mlir::isa<mlir::quir::Cbit_ExtractBitOp>(userOp))
          varUses.push_back(userOp);
      }

      for (Operation *assignment : varAssignments) {
        llvm::errs() << "  ";
        assignment->dump();

        for (Operation *use : varUses) {
          bool dominates = domInfo.dominates(assignment, use);
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
  PassRegistration<quir::TestPrintNestingPass>();
  PassRegistration<quir::FunctionArgumentSpecializationPass>();
  PassRegistration<quir::ClassicalOnlyDetectionPass>();
  PassRegistration<quir::BreakResetPass>();
  PassRegistration<quir::MergeParallelResetsPass>();
  PassRegistration<quir::SubroutineCloningPass>();
  PassRegistration<quir::RemoveQubitArgsPass>();
  PassRegistration<quir::UnusedVariablePass>();
  PassRegistration<quir::AddShotLoopPass>();
  PassRegistration<quir::QuantumDecorationPass>();
  PassRegistration<quir::MergeMeasuresPass>();
  PassRegistration<quir::QUIRAngleConversionPass>();
  PassRegistration<quir::LoadEliminationPass>();
  PassRegistration<quir::DumpVariableDominanceInfoPass>();
}

void registerQuirPassPipeline() {
  PassPipelineRegistration<> pipeline("quirOpt",
                                      "Enable QUIR-specific optimizations",
                                      quir::quirPassPipelineBuilder);
}
} // end namespace mlir::quir
