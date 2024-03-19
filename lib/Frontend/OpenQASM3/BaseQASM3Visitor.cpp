//===- BaseQASM3Visitor.cpp -------------------------------------*- C++ -*-===//
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
//  The BaseQASM3Visitor implements a simple visitor pattern to traverse the AST
//  which is output from the qss-qasm parser.
//
//  The BaseQASM3Visitor is a base class that cannot decide for itself what to
//  do upon visiting a node. The derived classes of BaseQASM3Visitor (e.g.
//  PrintQASM3Visitor, QUIRGenQASM3Visitor) will almost always determine what to
//  do with leaf nodes.
//
//  The implementation in this file is focused on extracting another node from a
//  given node, and then calling `visit` on it.
//
//  For instance, an ASTStatementNode could be of several different subtypes,
//  so we attempt to cast it to its specific type and follow with another call
//  to `visit` that can be properly dispatched, both by the argument type and
//  the derived class type (e.g. PrintQASM3Visitor or QUIRGenQASM3Visitor).
//
//  Development tips:
//
//  When adding support for a new feature, this process may be helpful:
//  1. Create an example file, which can later be used as a test case
//  2. Execute the compiler using the AST printer
//     (`./qss-compiler <your-input-file.qasm> --emit=ast`)
//  3. Inspect the output to discover the relevant AST nodes
//  4. Execute the compiler again using `--emit=mlir`. Often times, given the
//     existing code, this will print a nice error such as
//     "Cannot process ASTTypeExampleNode statement node.". This tells you
//     that the node you must add support for is a statement node of type
//     ASTTypeExampleNode.
//  5. Search within `qss-qasm` "class ASTExampleNode" to quickly find the
//     node's user interface. This interface will tell you how to visit nested
//     nodes, if there are any, and/or what to do with the nodes when they are
//     visited in the concrete implementations.
//  6. Update headers (BaseQASM3Visitor and its implementations) with the new
//  nodes
//     you've seen. Follow the patterns below to add support for them in
//     BaseQASM3Visitor, if necessary.
//  7. Add implementations in the derived classes.
//
//===----------------------------------------------------------------------===//

#include "Frontend/OpenQASM3/BaseQASM3Visitor.h"

#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <map>
#include <qasm/AST/ASTBarrier.h>
#include <qasm/AST/ASTBase.h>
#include <qasm/AST/ASTCBit.h>
#include <qasm/AST/ASTCastExpr.h>
#include <qasm/AST/ASTDeclarationList.h>
#include <qasm/AST/ASTDelay.h>
#include <qasm/AST/ASTDuration.h>
#include <qasm/AST/ASTExpression.h>
#include <qasm/AST/ASTFunctionCallExpr.h>
#include <qasm/AST/ASTFunctions.h>
#include <qasm/AST/ASTGates.h>
#include <qasm/AST/ASTIdentifier.h>
#include <qasm/AST/ASTIfConditionals.h>
#include <qasm/AST/ASTKernel.h>
#include <qasm/AST/ASTLoops.h>
#include <qasm/AST/ASTMeasure.h>
#include <qasm/AST/ASTParameterList.h>
#include <qasm/AST/ASTQubit.h>
#include <qasm/AST/ASTReset.h>
#include <qasm/AST/ASTResult.h>
#include <qasm/AST/ASTReturn.h>
#include <qasm/AST/ASTStatement.h>
#include <qasm/AST/ASTStretch.h>
#include <qasm/AST/ASTSwitchStatement.h>
#include <qasm/AST/ASTSymbolTable.h>
#include <qasm/AST/ASTTypeEnums.h>
#include <qasm/AST/ASTTypes.h>
#include <sstream>
#include <stdexcept>

using namespace QASM;

namespace qssc::frontend::openqasm3 {

BaseQASM3Visitor::~BaseQASM3Visitor() = default;

void BaseQASM3Visitor::setStatementList(ASTStatementList *list) {
  statementList = list;
}

void BaseQASM3Visitor::walkAST() { visit(statementList); }

void BaseQASM3Visitor::visit(const ASTStatementList *list) {
  for (ASTStatement *i : *list) {
    if (auto *declNode = dynamic_cast<ASTDeclarationNode *>(i)) {
      visit(declNode);
    } else if (auto *statementNode = dynamic_cast<ASTStatementNode *>(i)) {
      visit(statementNode);
    } else {
      throw std::runtime_error("Could not cast ASTStatement to "
                               "ASTDeclarationNode or ASTStatementNode.\n");
    }
  }
}

void BaseQASM3Visitor::visit(const ASTSymbolTableEntry *symTableEntry) {
  assert(symTableEntry);
  switch (ASTType const astType = symTableEntry->GetValueType()) {
  case ASTTypeQubitContainer:
    dispatchSymbolTableEntryVisit<ASTQubitContainerNode>(symTableEntry);
    break;

  case ASTTypeBitset:
    dispatchSymbolTableEntryVisit<ASTCBitNode>(symTableEntry);
    break;

  case ASTTypeDuration:
    dispatchSymbolTableEntryVisit<ASTDurationNode>(symTableEntry);
    break;

  case ASTTypeBool:
    dispatchSymbolTableEntryVisit<ASTBoolNode>(symTableEntry);
    break;

  case ASTTypeInt:
    dispatchSymbolTableEntryVisit<ASTIntNode>(symTableEntry);
    break;

  case ASTTypeMPInteger:
    dispatchSymbolTableEntryVisit<ASTMPIntegerNode>(symTableEntry);
    break;

  case ASTTypeFloat:
    dispatchSymbolTableEntryVisit<ASTFloatNode>(symTableEntry);
    break;

  case ASTTypeMPDecimal:
    dispatchSymbolTableEntryVisit<ASTMPDecimalNode>(symTableEntry);
    break;

  case ASTTypeMPComplex:
    dispatchSymbolTableEntryVisit<ASTMPComplexNode>(symTableEntry);
    break;

  case ASTTypeAngle:
    dispatchSymbolTableEntryVisit<ASTAngleNode>(symTableEntry);
    break;

  case ASTTypeKernel:
    dispatchSymbolTableEntryVisit<ASTKernelNode>(symTableEntry);
    break;

  case ASTTypeGateQubitParam:
    dispatchSymbolTableEntryVisit<ASTQubitNode>(symTableEntry);
    break;

  default: {
    std::ostringstream oss;
    oss << "Cannot process " << PrintTypeEnum(astType)
        << " declaration node.\n";
    throw std::runtime_error(oss.str());
  }
  }
}

void BaseQASM3Visitor::visit(const ASTStatementNode *node) {
  // evaluate statements by type
  switch (ASTType const astType = node->GetASTType()) {
  case ASTTypeOpenQASMStatement:
    dispatchVisit<ASTOpenQASMStatementNode>(node);
    break;

  case ASTTypeForStatement:
    dispatchVisit<ASTForStatementNode>(node);
    break;

  case ASTTypeForLoop:
    dispatchVisit<ASTForLoopNode>(node);
    break;

  case ASTTypeIfStatement:
    dispatchVisit<ASTIfStatementNode>(node);
    break;

  case ASTTypeElseStatement:
    dispatchVisit<ASTElseStatementNode>(node);
    break;

  case ASTTypeSwitchStatement:
    dispatchVisit<ASTSwitchStatementNode>(node);
    break;

  case ASTTypeWhileStatement:
    dispatchVisit<ASTWhileStatementNode>(node);
    break;

  case ASTTypeWhileLoop:
    dispatchVisit<ASTWhileLoopNode>(node);
    break;

  case ASTTypeReturn:
    dispatchVisit<ASTReturnStatementNode>(node);
    break;

  case ASTTypeGateDeclaration:
    dispatchVisit<ASTGateDeclarationNode>(node);
    break;

  case ASTTypeGateGenericOpNode:
    dispatchVisit<ASTGenericGateOpNode>(node);
    break;

  case ASTTypeGateHOpNode:
    dispatchVisit<ASTHGateOpNode>(node);
    break;

  case ASTTypeGateUOpNode:
    dispatchVisit<ASTUGateOpNode>(node);
    break;

  case ASTTypeCXGateOpNode:
    dispatchVisit<ASTCXGateOpNode>(node);
    break;

  case ASTTypeReset:
    dispatchVisit<ASTResetNode>(node);
    break;

  case ASTTypeMeasure:
    dispatchVisit<ASTMeasureNode>(node);
    break;

  case ASTTypeDelayStatement:
    dispatchVisit<ASTDelayStatementNode>(node);
    break;

  case ASTTypeBarrier:
    dispatchVisit<ASTBarrierNode>(node);
    break;

  case ASTTypeStretchStatement:
    dispatchVisit<ASTStretchStatementNode>(node);
    break;

  case ASTTypeBinaryOpStatement:
    dispatchVisit<ASTBinaryOpStatementNode>(node);
    break;

  default:
    std::ostringstream oss;
    oss << "Cannot process " << PrintTypeEnum(astType) << " statement node.\n";
    throw std::runtime_error(oss.str());
  }
}

void BaseQASM3Visitor::visit(const ASTBinaryOpStatementNode *node) {
  visit(node->GetBinaryOp());
}

void BaseQASM3Visitor::visit(const ASTOpenQASMStatementNode *node) {}

void BaseQASM3Visitor::visit(const ASTExpressionNode *node) {
  switch (ASTType const astType = node->GetASTType()) {
  case ASTTypeIdentifier: {
    const ASTIdentifierNode *identifierNode = nullptr;
    if (node->IsIdentifier()) {
      identifierNode = node->GetIdentifier();
    } else if (node->IsExpression()) {
      identifierNode =
          dynamic_cast<const ASTIdentifierNode *>(node->GetExpression());
    }
    assert(identifierNode &&
           "Invalid ASTIdentifierNode from ASTExpressionNode!");

    if (identifierNode->IsReference())
      dispatchVisit<ASTIdentifierRefNode>(node->GetExpression());
    else
      visit(identifierNode);
    break;
  }
  case ASTTypeInt:
    dispatchVisit<ASTIntNode>(node);
    break;

  case ASTTypeFloat:
    dispatchVisit<ASTFloatNode>(node);
    break;

  case ASTTypeMPDecimal:
    dispatchVisit<ASTMPDecimalNode>(node);
    break;

  case ASTTypeBinaryOp:
    dispatchVisit<ASTBinaryOpNode>(node);
    break;

  case ASTTypeBool:
    dispatchVisit<ASTBoolNode>(node);
    break;

  case ASTTypeFunction:
    dispatchVisit<ASTFunctionDefinitionNode>(node);
    break;

  case ASTTypeFunctionCall:
    dispatchVisit<ASTFunctionCallNode>(node);
    break;

  case ASTTypeCast:
    dispatchVisit<ASTCastExpressionNode>(node);
    break;

  case ASTTypeBitset:
    dispatchVisit<ASTCBitNode>(node);
    break;

  case ASTTypeMPInteger:
    dispatchVisit<ASTMPIntegerNode>(node);
    break;

  case ASTTypeAngle:
    dispatchVisit<ASTAngleNode>(node);
    break;

  case ASTTypeOpTy:
    dispatchVisit<ASTOperatorNode>(node);
    break;

  case ASTTypeOpndTy:
    dispatchVisit<ASTOperandNode>(node);
    break;

  case ASTTypeUnaryOp:
    dispatchVisit<ASTUnaryOpNode>(node);
    break;

  case ASTTypeMPComplex:
    dispatchVisit<ASTMPComplexNode>(node);
    break;

  default:
    std::ostringstream oss;
    oss << "Cannot process " << PrintTypeEnum(astType) << " expression node.\n";
    throw std::runtime_error(oss.str());
  }
}

void BaseQASM3Visitor::visit(const ASTCastExpressionNode *node) {
  switch (node->GetCastFrom()) {
  case ASTTypeAngle:
    visit(node->GetAngle());
    break;
  case ASTTypeBinaryOp:
    visit(node->GetBinaryOp());
    break;
  case ASTTypeBitset:
    visit(node->GetCBit());
    break;
  case ASTTypeBool:
    visit(node->GetBool());
    break;
  case ASTTypeDouble:
    visit(node->GetDouble());
    break;
  case ASTTypeInt:
    visit(node->GetInt());
    break;
  case ASTTypeFloat:
    visit(node->GetFloat());
    break;
  case ASTTypeMPDecimal:
    visit(node->GetMPDecimal());
    break;
  case ASTTypeMPInteger:
    visit(node->GetMPInteger());
    break;
  case ASTTypeUnaryOp:
    visit(node->GetUnaryOp());
    break;
  case ASTTypeIdentifier:
    visit(node->GetTargetIdentifier());
    break;

  default:
    llvm::errs() << "Unhandled source type "
                 << PrintTypeEnum(node->GetCastFrom()) << "\n";
    llvm_unreachable("unhandled source type in ASTCastExpressionNode");
  }
}

void BaseQASM3Visitor::visit(const ASTKernelNode *node) {
  visit(&node->GetParameters());
  visit(node->GetResult());
}

void BaseQASM3Visitor::visit(const ASTParameterList *list) {
  visit(list->GetDeclarationList());
}

void BaseQASM3Visitor::visit(
    const std::map<unsigned, ASTDeclarationNode *> *map) {
  for (auto elem : *map)
    visit(elem.second);
}

void BaseQASM3Visitor::visit(const ASTDeclarationList *list) {
  for (ASTDeclarationNode *declNode : *list)
    visit(declNode);
}

void BaseQASM3Visitor::visit(const ASTIdentifierList *list) {
  for (ASTIdentifierNode *identNode : *list)
    visit(identNode);
}

void BaseQASM3Visitor::visit(const ASTExpressionList *list) {
  for (ASTBase *node : *list) {
    switch (node->GetASTType()) {
    case ASTTypeExpression:
      dispatchVisit<ASTExpressionNode>(node);
      break;
    case ASTTypeIdentifier:
      dispatchVisit<ASTExpressionNode>(node);
      break;
    default:
      llvm::errs() << "Unhandled source type "
                   << PrintTypeEnum(node->GetASTType()) << "\n";
      llvm_unreachable("unhandled source type in ASTExpressionList");
    }
  }
}

void BaseQASM3Visitor::visit(const ASTResultNode *node) {
  // If no result is present visitation is over

  if (!node->HasResult())
    return;

  // If a result is present, continue visitation
  switch (node->GetResultType()) {
  case ASTTypeAngle:
    visit(node->GetAngleNode());
    break;
  case ASTTypeBool:
    visit(node->GetBoolNode());
    break;
  case ASTTypeBitset:
    visit(node->GetCBitNode());
    break;
  case ASTTypeInt:
    visit(node->GetIntNode());
    break;
  case ASTTypeFloat:
    visit(node->GetFloatNode());
    break;
  case ASTTypeMPInteger:
    visit(node->GetMPInteger());
    break;
  case ASTTypeMPDecimal:
    visit(node->GetMPDecimal());
    break;
  case ASTTypeMPComplex:
    visit(node->GetMPDecimal());
    break;
  case ASTTypeVoid:
    visit(node->GetVoidNode());
    break;

  default:
    llvm::errs() << "Unhandled result type "
                 << PrintTypeEnum(node->GetResultType()) << "\n";
    llvm_unreachable("unhandled result type in ASTResultNode");
  }
}

void BaseQASM3Visitor::visit(const ASTFunctionCallNode *node) {
  switch (node->GetFunctionCallType()) {
  case ASTTypeFunctionCallExpression:
    visit(&node->GetExpressionList());
    visit(&node->GetQuantumIdentifierList());
    break;
  case ASTTypeKernelCallExpression:
    visit(&node->GetExpressionList());
    visit(node->GetKernelDefinition());
    break;
  default:
    llvm::errs() << "Unhandled call type "
                 << PrintTypeEnum(node->GetFunctionCallType()) << "\n";
    llvm_unreachable("unhandled call type in ASTFunctionCallNode");
  }
}

} // namespace qssc::frontend::openqasm3
