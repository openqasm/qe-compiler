//===- PrintQASM3Visitor.h --------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021 - 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

  void visit(const QASM::ASTUnaryOpNode *) override;
};

} // namespace qssc::frontend::openqasm3

#endif // VISITOR_PRINT_VISITOR_H
