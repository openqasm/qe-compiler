//===- BaseQASM3Visitor.h ---------------------------------------*- C++ -*-===//
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

#ifndef VISITOR_BASE_VISITOR_H
#define VISITOR_BASE_VISITOR_H

#include <qasm/AST/AST.h>
#include <qasm/AST/ASTBarrier.h>
#include <qasm/AST/ASTCastExpr.h>
#include <qasm/AST/ASTDelay.h>
#include <qasm/AST/ASTFunctionCallExpr.h>
#include <qasm/AST/ASTIfConditionals.h>
#include <qasm/AST/ASTIntegerList.h>
#include <qasm/AST/ASTLoops.h>
#include <qasm/AST/ASTStretch.h>
#include <qasm/AST/ASTSwitchStatement.h>
#include <qasm/AST/ASTSymbolTable.h>
#include <qasm/AST/ASTValue.h>

#include "mlir/Support/LogicalResult.h"

#include <string>

namespace qssc::frontend::openqasm3 {

class BaseQASM3Visitor {

protected:
  QASM::ASTStatementList *statementList;

public:
  BaseQASM3Visitor(QASM::ASTStatementList *sList) : statementList(sList) {}

  BaseQASM3Visitor() = default;

  virtual ~BaseQASM3Visitor() = 0;

  void setStatementList(QASM::ASTStatementList *);

  void walkAST();

  template <typename T>
  void dispatchVisit(const QASM::ASTBase *node) {
    const T *specificNode = dynamic_cast<const T *>(node);
    assert(specificNode && "Could not cast ASTNode to specific node type");
    visit(specificNode);
  }

  template <typename T>
  void dispatchSymbolTableEntryVisit(const QASM::ASTSymbolTableEntry *entry) {
    const T *specificValue = entry->GetValue()->GetValue<T *>();
    assert(specificValue && "Could not cast with GetValue to specific type");
    visit(specificValue);
  }

  void visit(const QASM::ASTStatementList *);

  void visit(const QASM::ASTSymbolTableEntry *);

  void visit(const QASM::ASTStatementNode *);

  void visit(const QASM::ASTOpenQASMStatementNode *);

  void visit(const QASM::ASTExpressionNode *);

  void visit(const QASM::ASTBinaryOpStatementNode *);

  virtual void visit(const QASM::ASTForStatementNode *) = 0;

  virtual void visit(const QASM::ASTForLoopNode *) = 0;

  virtual void visit(const QASM::ASTIfStatementNode *) = 0;

  virtual void visit(const QASM::ASTElseStatementNode *) = 0;

  virtual void visit(const QASM::ASTSwitchStatementNode *) = 0;

  virtual void visit(const QASM::ASTWhileStatementNode *) = 0;

  virtual void visit(const QASM::ASTWhileLoopNode *) = 0;

  virtual void visit(const QASM::ASTReturnStatementNode *) = 0;

  virtual void visit(const QASM::ASTResultNode *);

  virtual void visit(const QASM::ASTFunctionDeclarationNode *) = 0;

  virtual void visit(const QASM::ASTFunctionDefinitionNode *) = 0;

  virtual void visit(const QASM::ASTGateDeclarationNode *) = 0;

  virtual void visit(const QASM::ASTGenericGateOpNode *) = 0;

  virtual void visit(const QASM::ASTGateNode *) = 0;

  virtual void visit(const QASM::ASTHGateOpNode *) = 0;

  virtual void visit(const QASM::ASTUGateOpNode *) = 0;

  virtual void visit(const QASM::ASTCXGateOpNode *) = 0;

  virtual void visit(const QASM::ASTResetNode *) = 0;

  virtual void visit(const QASM::ASTMeasureNode *) = 0;

  virtual void visit(const QASM::ASTDelayStatementNode *) = 0;

  virtual void visit(const QASM::ASTDelayNode *) = 0;

  virtual void visit(const QASM::ASTBarrierNode *) = 0;

  virtual void visit(const QASM::ASTDeclarationNode *) = 0;

  virtual void visit(const QASM::ASTKernelDeclarationNode *) = 0;

  virtual void visit(const QASM::ASTQubitContainerNode *) = 0;

  virtual void visit(const QASM::ASTQubitNode *) = 0;

  virtual void visit(const QASM::ASTCBitNode *) = 0;

  virtual void visit(const QASM::ASTDurationNode *) = 0;

  virtual void visit(const QASM::ASTStretchStatementNode *) = 0;

  virtual void visit(const QASM::ASTStretchNode *) = 0;

  virtual void visit(const QASM::ASTIdentifierRefNode *) = 0;

  virtual void visit(const QASM::ASTIdentifierNode *) = 0;

  virtual void visit(const QASM::ASTBinaryOpNode *) = 0;

  virtual void visit(const QASM::ASTIntNode *) = 0;

  virtual void visit(const QASM::ASTMPIntegerNode *) = 0;

  virtual void visit(const QASM::ASTFloatNode *) = 0;

  virtual void visit(const QASM::ASTMPDecimalNode *) = 0;

  virtual void visit(const QASM::ASTMPComplexNode *) = 0;

  virtual void visit(const QASM::ASTAngleNode *) = 0;

  virtual void visit(const QASM::ASTBoolNode *) = 0;

  virtual void visit(const QASM::ASTCastExpressionNode *);

  virtual void visit(const QASM::ASTKernelNode *);

  virtual void visit(const QASM::ASTParameterList *);

  virtual void visit(const std::map<unsigned, QASM::ASTDeclarationNode *> *);

  virtual void visit(const QASM::ASTDeclarationList *);

  virtual void visit(const QASM::ASTExpressionList *);

  virtual void visit(const QASM::ASTIdentifierList *);

  virtual void visit(const QASM::ASTFunctionCallNode *);

  virtual void visit(const QASM::ASTVoidNode *) = 0;

  virtual void visit(const QASM::ASTOperatorNode *) = 0;

  virtual void visit(const QASM::ASTOperandNode *) = 0;

  virtual void visit(const QASM::ASTUnaryOpNode *) = 0;
};

} // namespace qssc::frontend::openqasm3

#endif // VISITOR_BASE_VISITOR_H
