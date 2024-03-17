//===- PrintQASM3Visitor.h --------------------------------------*- C++ -*-===//
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

#ifndef VISITOR_PRINT_VISITOR_H
#define VISITOR_PRINT_VISITOR_H

#include "Frontend/OpenQASM3/BaseQASM3Visitor.h"

namespace qssc::frontend::openqasm3 {

class PrintQASM3Visitor : public BaseQASM3Visitor {
private:
  std::ostream &vStream; // visitor output stream

public:
  PrintQASM3Visitor(QASM::ASTStatementList *sList, std::ostream &os)
      : BaseQASM3Visitor(sList), vStream(os) {}

  PrintQASM3Visitor(std::ostream &os) : BaseQASM3Visitor(), vStream(os) {}

  void visit(const QASM::ASTForStatementNode *) override;

  void visit(const QASM::ASTForLoopNode *) override;

  void visit(const QASM::ASTIfStatementNode *) override;

  void visit(const QASM::ASTElseStatementNode *) override;

  void visit(const QASM::ASTSwitchStatementNode *) override;

  void visit(const QASM::ASTWhileStatementNode *) override;

  void visit(const QASM::ASTWhileLoopNode *) override;

  void visit(const QASM::ASTReturnStatementNode *) override;

  void visit(const QASM::ASTResultNode *) override;

  void visit(const QASM::ASTFunctionDeclarationNode *) override;

  void visit(const QASM::ASTFunctionDefinitionNode *) override;

  void visit(const QASM::ASTGateDeclarationNode *) override;

  void visit(const QASM::ASTGenericGateOpNode *) override;

  void visit(const QASM::ASTGateNode *) override;

  void visit(const QASM::ASTHGateOpNode *) override;

  void visit(const QASM::ASTUGateOpNode *) override;

  void visit(const QASM::ASTCXGateOpNode *) override;

  void visit(const QASM::ASTResetNode *) override;

  void visit(const QASM::ASTMeasureNode *) override;

  void visit(const QASM::ASTDelayStatementNode *) override;

  void visit(const QASM::ASTDelayNode *) override;

  void visit(const QASM::ASTBarrierNode *) override;

  void visit(const QASM::ASTDeclarationNode *) override;

  void visit(const QASM::ASTKernelDeclarationNode *) override;

  void visit(const QASM::ASTQubitContainerNode *) override;

  void visit(const QASM::ASTQubitNode *) override;

  void visit(const QASM::ASTCBitNode *) override;

  void visit(const QASM::ASTDurationNode *) override;

  void visit(const QASM::ASTStretchStatementNode *) override;

  void visit(const QASM::ASTStretchNode *) override;

  void visit(const QASM::ASTIdentifierRefNode *) override;

  void visit(const QASM::ASTIdentifierNode *) override;

  void visit(const QASM::ASTBinaryOpNode *) override;

  void visit(const QASM::ASTIntNode *) override;

  void visit(const QASM::ASTMPIntegerNode *) override;

  void visit(const QASM::ASTFloatNode *) override;

  void visit(const QASM::ASTMPDecimalNode *) override;

  void visit(const QASM::ASTMPComplexNode *) override;

  void visit(const QASM::ASTAngleNode *) override;

  void visit(const QASM::ASTBoolNode *) override;

  void visit(const QASM::ASTCastExpressionNode *) override;

  void visit(const QASM::ASTKernelNode *) override;

  void visit(const QASM::ASTDeclarationList *list) override;

  void visit(const QASM::ASTFunctionCallNode *node) override;

  void visit(const QASM::ASTVoidNode *) override;

  void visit(const QASM::ASTOperatorNode *) override;

  void visit(const QASM::ASTOperandNode *) override;

  void visit(const QASM::ASTUnaryOpNode *) override;
};

} // namespace qssc::frontend::openqasm3

#endif // VISITOR_PRINT_VISITOR_H
