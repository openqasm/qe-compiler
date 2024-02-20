//===- QUIRGenQASM3Visitor.h ------------------------------------*- C++ -*-===//
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

#ifndef VISITOR_QUIR_GEN_VISITOR_H
#define VISITOR_QUIR_GEN_VISITOR_H

#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIREnums.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"
#include "Dialect/QUIR/Transforms/Passes.h"
#include "Frontend/OpenQASM3/BaseQASM3Visitor.h"
#include "Frontend/OpenQASM3/QUIRVariableBuilder.h"

#include "mlir/Dialect/Complex/IR/Complex.h"
#include "llvm/Support/Error.h"

#include <unordered_map>

namespace qssc::frontend::openqasm3 {

class QUIRGenQASM3Visitor : public BaseQASM3Visitor {
private:
  // References to MLIR single static assignment Values
  // (TODO needs to be refactored)
  std::unordered_map<std::string, mlir::Value> ssaValues;
  std::vector<mlir::Value> ssaOtherValues;
  mlir::OpBuilder builder;
  mlir::OpBuilder topLevelBuilder;
  mlir::OpBuilder circuitParentBuilder;
  mlir::ModuleOp newModule;
  mlir::quir::CircuitOp currentCircuitOp;
  std::string filename;
  bool hasFailed{false};
  bool buildingInCircuit{false};
  uint circuitCount{0};

  mlir::Location getLocation(const QASM::ASTBase *);
  bool assign(mlir::Value &, const std::string &);
  mlir::Value getCurrentValue(const std::string &valueName);
  llvm::Expected<std::string>
  getExpressionName(const QASM::ASTExpressionNode *);

  /// hold intermediate expression value while visiting child nodes
  llvm::Expected<mlir::Value> expression;

  llvm::Expected<mlir::Value>
  visitAndGetExpressionValue(QASM::ASTExpressionNode const *node);
  template <class NodeType>
  llvm::Expected<mlir::Value> visitAndGetExpressionValue(NodeType const *node) {
    return visit_(node);
  }

  mlir::Value createDurationRef(const mlir::Location &, uint64_t,
                                const QASM::LengthUnit &);

  QUIRVariableBuilder varHandler;

  mlir::Value createVoidValue(mlir::Location);
  mlir::Value createVoidValue(QASM::ASTBase const *node);

  mlir::Type getCastDestinationType(const QASM::ASTCastExpressionNode *node,
                                    mlir::OpBuilder &builder);

  llvm::Expected<std::string> resolveQCParam(const QASM::ASTGateNode *gateNode,
                                             unsigned int index);

  /// \brief
  /// Create a diagnostic with the specified severity and location from the
  /// provided AST node.
  ///
  /// If the severity is of level Error, mark the run of QUIRGenQASM3Visitor as
  /// failed. This function returns an in-flight diagnostic that can be used
  /// like a C++ output stream to add messages.
  ///
  /// \param location location of the diagnostic
  /// \param severity severity of the diagnostic
  ///
  /// \returns an in-flight diagnostic that allows adding messages and notes.
  mlir::InFlightDiagnostic reportError(QASM::ASTBase const *location,
                                       mlir::DiagnosticSeverity severity);

public:
  QUIRGenQASM3Visitor(QASM::ASTStatementList *sList, mlir::OpBuilder b,
                      mlir::ModuleOp newModule, std::string f)
      : BaseQASM3Visitor(sList), builder(b), topLevelBuilder(b),
        circuitParentBuilder(b), newModule(newModule), filename(std::move(f)),
        expression(llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "Testing error")),
        varHandler(builder) {}

  QUIRGenQASM3Visitor(mlir::OpBuilder b, mlir::ModuleOp newModule,
                      std::string filename)
      : builder(b), topLevelBuilder(b), circuitParentBuilder(b),
        newModule(newModule), filename(std::move(filename)),
        expression(llvm::createStringError(llvm::inconvertibleErrorCode(),
                                           "Testing error")),
        varHandler(builder) {}

  void initialize(uint numShots, const double &shotDelay,
                  const mlir::quir::TimeUnits &shotDelayUnits);

  void startCircuit(mlir::Location location);
  void finishCircuit();
  void switchCircuit(bool buildInCircuit, mlir::Location location);

  void setInputFile(std::string);

  mlir::LogicalResult walkAST();

protected:
  using ExpressionValueType = llvm::Expected<mlir::Value>;

  void visit(const QASM::ASTForStatementNode *) override;

  void visit(const QASM::ASTForLoopNode *) override;

  void visit(const QASM::ASTIfStatementNode *) override;

  void visit(const QASM::ASTElseStatementNode *) override;

  void visit(const QASM::ASTSwitchStatementNode *) override;

  void visit(const QASM::ASTWhileStatementNode *) override;

  void visit(const QASM::ASTWhileLoopNode *) override;

  void visit(const QASM::ASTReturnStatementNode *) override;

  template <typename NodeType>
  void visitWithReturn(NodeType *node) {
    expression = visit_(node);
  }

  void visit(const QASM::ASTResultNode *node) override {
    visitWithReturn(node);
  }
  ExpressionValueType visit_(const QASM::ASTResultNode *);

  void visit(const QASM::ASTFunctionDeclarationNode *) override;

  void visit(const QASM::ASTFunctionDefinitionNode *) override;

  void visit(const QASM::ASTFunctionCallNode *) override;

  void visit(const QASM::ASTGateDeclarationNode *) override;

  void visit(const QASM::ASTGenericGateOpNode *) override;

  void visit(const QASM::ASTGateNode *node) override { visitWithReturn(node); }
  ExpressionValueType visit_(const QASM::ASTGateNode *node);

  void visit(const QASM::ASTHGateOpNode *) override;

  void visit(const QASM::ASTUGateOpNode *) override;

  void visit(const QASM::ASTCXGateOpNode *) override;

  void visit(const QASM::ASTResetNode *) override;

  mlir::Value createMeasurement(const QASM::ASTMeasureNode *node,
                                bool emitAssignment);
  void visit(const QASM::ASTMeasureNode *) override;

  void visit(const QASM::ASTDelayStatementNode *) override;

  void visit(const QASM::ASTDelayNode *) override {} /* unused */

  void visit(const QASM::ASTBarrierNode *) override;

  void visit(const QASM::ASTDeclarationNode *) override;

  void visit(const QASM::ASTKernelDeclarationNode *) override;

  void visit(const QASM::ASTKernelNode *) override;

  void visit(const QASM::ASTQubitContainerNode *node) override {
    visitWithReturn(node);
  };
  ExpressionValueType visit_(const QASM::ASTQubitContainerNode *);

  void visit(const QASM::ASTQubitNode *node) override {
    visitWithReturn(node);
  };
  ExpressionValueType visit_(const QASM::ASTQubitNode *);

  void visit(const QASM::ASTCBitNode *node) override { visitWithReturn(node); };
  ExpressionValueType visit_(const QASM::ASTCBitNode *node);

  void visit(const QASM::ASTDurationNode *node) override {
    visitWithReturn(node);
  };
  ExpressionValueType visit_(const QASM::ASTDurationNode *node);

  void visit(const QASM::ASTStretchStatementNode *) override;
  ExpressionValueType visit_(const QASM::ASTStretchStatementNode *);

  void visit(const QASM::ASTStretchNode *node) override {
    visitWithReturn(node);
  };
  ExpressionValueType visit_(const QASM::ASTStretchNode *node);

  void visit(const QASM::ASTIdentifierRefNode *node) override {
    visitWithReturn(node);
  };
  ExpressionValueType visit_(const QASM::ASTIdentifierRefNode *node);

  void visit(const QASM::ASTIdentifierNode *node) override {
    visitWithReturn(node);
  };
  ExpressionValueType visit_(const QASM::ASTIdentifierNode *node);

  void visit(const QASM::ASTBinaryOpNode *node) override {
    visitWithReturn(node);
  };
  ExpressionValueType visit_(const QASM::ASTBinaryOpNode *node);

  void visit(const QASM::ASTIntNode *node) override { visitWithReturn(node); };
  ExpressionValueType visit_(const QASM::ASTIntNode *node);

  void visit(const QASM::ASTMPIntegerNode *node) override {
    visitWithReturn(node);
  };
  ExpressionValueType visit_(const QASM::ASTMPIntegerNode *node);

  void visit(const QASM::ASTFloatNode *node) override {
    visitWithReturn(node);
  };
  ExpressionValueType visit_(const QASM::ASTFloatNode *node);

  void visit(const QASM::ASTMPDecimalNode *node) override {
    visitWithReturn(node);
  };
  ExpressionValueType visit_(const QASM::ASTMPDecimalNode *node);

  void visit(const QASM::ASTMPComplexNode *node) override {
    visitWithReturn(node);
  };
  ExpressionValueType visit_(const QASM::ASTMPComplexNode *node);

  void visit(const QASM::ASTAngleNode *node) override {
    visitWithReturn(node);
  };
  ExpressionValueType visit_(const QASM::ASTAngleNode *node);

  void visit(const QASM::ASTBoolNode *node) override { visitWithReturn(node); };
  ExpressionValueType visit_(const QASM::ASTBoolNode *node);

  void visit(const QASM::ASTCastExpressionNode *node) override {
    visitWithReturn(node);
  };
  ExpressionValueType visit_(const QASM::ASTCastExpressionNode *node);

  void visit(const QASM::ASTVoidNode *) override {}

  void visit(const QASM::ASTOperatorNode *node) override {
    visitWithReturn(node);
  }
  ExpressionValueType visit_(const QASM::ASTOperatorNode *) {
    llvm::report_fatal_error("not to be used with a visit function yet");
  }

  void visit(const QASM::ASTOperandNode *node) override {
    visitWithReturn(node);
  }
  ExpressionValueType visit_(const QASM::ASTOperandNode *) {
    llvm::report_fatal_error("not to be used with a visit function yet");
  }

  void visit(const QASM::ASTUnaryOpNode *node) override {
    visitWithReturn(node);
  }
  ExpressionValueType visit_(const QASM::ASTUnaryOpNode *);

private:
  ExpressionValueType handleAssign(const QASM::ASTBinaryOpNode *);
  ExpressionValueType handleAssign(const QASM::ASTBinaryOpNode *, mlir::Value);

  ExpressionValueType getValueFromLiteral(const QASM::ASTMPDecimalNode *);

  mlir::Type getQUIRTypeFromDeclaration(const QASM::ASTDeclarationNode *);
};

} // namespace qssc::frontend::openqasm3

#endif // VISITOR_QUIR_GEN_VISITOR_H
