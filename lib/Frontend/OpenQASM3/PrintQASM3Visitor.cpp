//===- PrintQASM3Visitor.cpp ------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
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
//
//  The PrintQASM3Visitor will print a concise version of the AST generated by
//  the qss-qasm parser.
//
//  Generally, it prints in the format: NodeName(attribute0, attribute1, ...)
//  - When applicable, the first attribute should be the name of the variable/
//    function/gate, etc.
//  - If a node contains another node, it's usually preferred to visit that node
//    and complete the implementation in another visitor method rather than
//    extracting nested information.
//
//===----------------------------------------------------------------------===//

#include "Frontend/OpenQASM3/PrintQASM3Visitor.h"

#include "qasm/AST/ASTIdentifier.h"

#include "llvm/Support/raw_ostream.h"

using namespace QASM;

namespace qssc::frontend::openqasm3 {

void PrintQASM3Visitor::visit(const ASTForStatementNode *node) {
  const ASTForLoopNode *loop = node->GetLoop();
  vStream << "ForStatementNode(";
  visit(loop);
  vStream << ")\n";
}

void PrintQASM3Visitor::visit(const ASTForLoopNode *node) {
  const ASTIntegerList &intList = node->GetIntegerList();

  if (node->GetIVMethod() == ASTForLoopNode::IVMethod::IVMonotonic) {
    if (intList.Size() == 2) {
      vStream << "start=" << intList.front() << ", end=" << intList.back();
    } else if (intList.Size() == 3) {
      vStream << "start=" << intList.front()
              << ", stepping=" << node->GetStepping()
              << ", end=" << intList.back();
    }
  } else {
    vStream << "i=[";
    for (int I : intList)
      vStream << I << " ";
    vStream << "], stepping=" << node->GetStepping();
  }
  vStream << ",\nstatements=\n";
  const ASTStatementList &loopNode = node->GetStatementList();
  BaseQASM3Visitor::visit(&loopNode);
}

void PrintQASM3Visitor::visit(const ASTIfStatementNode *node) {
  const ASTExpressionNode *exprNode = node->GetExpression();
  vStream << "\nIfStatementNode("
          << "\ncondition=";
  BaseQASM3Visitor::visit(exprNode);
  vStream << ",\nstatements=\n";
  // single statement within the if block
  if (const ASTStatementNode *opNode = node->GetOpNode())
    BaseQASM3Visitor::visit(opNode);

  // multiple statements within the if block
  if (const ASTStatementList *opList = node->GetOpList())
    BaseQASM3Visitor::visit(opList);
  vStream << ")\n";
}

void PrintQASM3Visitor::visit(const ASTElseStatementNode *node) {
  vStream << "ElseStatementNode("
          << "\nstatements=\n";
  // single statement within the else block
  if (const ASTStatementNode *opNode = node->GetOpNode())
    BaseQASM3Visitor::visit(opNode);

  // multiple statements within the else block
  if (const ASTStatementList *opList = node->GetOpList())
    BaseQASM3Visitor::visit(opList);
  vStream << ")\n";
}

void PrintQASM3Visitor::visit(const ASTSwitchStatementNode *node) {
  ASTType quantityType = node->GetQuantityType();
  vStream << "SwitchStatementNode("
          << "SwitchQuantity(";

  switch (quantityType) {
  case ASTTypeInt:
  case ASTTypeUInt:
    vStream << "name=" << node->GetIntQuantity()->GetName();
    break;
  case ASTTypeMPInteger:
    vStream << "name=" << node->GetMPIntegerQuantity()->GetName();
    break;
  case ASTTypeBinaryOp:
    vStream << "name=" << node->GetBinaryOpQuantity()->GetName();
    break;
  case ASTTypeUnaryOp:
    vStream << "name=" << node->GetUnaryOpQuantity()->GetName();
    break;
  case ASTTypeIdentifier:
    vStream << "name=" << node->GetIdentifierQuantity()->GetName();
    break;
  case ASTTypeFunctionCall:
    vStream << "name=" << node->GetFunctionCallQuantity()->GetName();
    break;
  default:
    break;
  }
  vStream << ", type=" << PrintTypeEnum(quantityType) << "),\n";
  vStream << "statements=[\n";
  for (auto const &[key, caseStatement] : node->GetCaseStatementsMap()) {
    vStream << "CaseStatementNode(";
    vStream << "case=" << caseStatement->GetCaseIndex() << ", ";
    const ASTStatementList *statementList = caseStatement->GetStatementList();
    BaseQASM3Visitor::visit(statementList);
    vStream << "),\n";
  }
  vStream << "],\n";
  vStream << "default statement=[\n";
  const ASTStatementList *statementList =
      node->GetDefaultStatement()->GetStatementList();
  BaseQASM3Visitor::visit(statementList);
  vStream << "])\n";
}

void PrintQASM3Visitor::visit(const ASTWhileStatementNode *node) {
  const ASTWhileLoopNode *loop = node->GetLoop();
  vStream << "WhileStatement(";
  visit(loop);
  vStream << ")\n";
}

void PrintQASM3Visitor::visit(const ASTWhileLoopNode *node) {
  const ASTExpressionNode *condition = node->GetExpression();
  vStream << "condition=";
  BaseQASM3Visitor::visit(condition);
  vStream << ",\nstatements=\n";
  const ASTStatementList &loopNode = node->GetStatementList();
  BaseQASM3Visitor::visit(&loopNode);
}

void PrintQASM3Visitor::visit(const ASTReturnStatementNode *node) {
  vStream << "ReturnNode(";
  if (node->IsVoid()) {
    vStream << "void";
  } else if (const auto *nodeIdent = node->GetIdent()) {
    visit(nodeIdent);
  } else if (const auto *nodeInt = node->GetInt()) {
    visit(nodeInt);
  } else if (const auto *nodeInteger = node->GetMPInteger()) {
    visit(nodeInteger);
  } else if (const auto *nodeDecimal = node->GetMPDecimal()) {
    visit(nodeDecimal);
  } else if (const auto *angle = node->GetAngle()) {
    visit(angle);
  } else if (const auto *cbit = node->GetCBit()) {
    visit(cbit);
  } else if (const auto *qubit = node->GetQubit()) {
    visit(qubit);
  } else if (const auto *measure = node->GetMeasure()) {
    visit(measure);
  } else if (const auto *binOp = node->GetBinaryOp()) {
    visit(binOp);
  } else if (const auto *expr = node->GetExpression()) {
    BaseQASM3Visitor::visit(expr);
  } else if (const auto *stmt = node->GetStatement()) {
    BaseQASM3Visitor::visit(stmt);
  } else {
    std::ostringstream oss;
    oss << "Cannot process return statement.\n";
    throw std::runtime_error(oss.str());
  }
  vStream << ")";
}

void PrintQASM3Visitor::visit(const ASTResultNode *node) {
  vStream << "ResultNode(";
  BaseQASM3Visitor::visit(node);
  vStream << ")";
}

void PrintQASM3Visitor::visit(const ASTFunctionDeclarationNode *node) {
  vStream << "FunctionDeclarationNode(";
  visit(node->GetDefinition());
  vStream << ")\n";
}

void PrintQASM3Visitor::visit(const ASTFunctionDefinitionNode *node) {
  vStream << "FunctionDefinitionNode(name=";
  vStream << node->GetName();
  vStream << ", mangled name=";
  vStream << node->GetMangledName();
  if (node->IsBuiltin())
    vStream << ", builtin=1";
  if (node->IsExtern())
    vStream << ", extern=1";
  if (node->HasEllipsis())
    vStream << ", has ellipsis=1";
  if (node->HasParameters()) {
    vStream << ",\nparameters=[";
    const auto &idxParamsMap = node->GetParameters();
    for (auto param : idxParamsMap)
      visit(param.second);
    vStream << "]";
  }
  if (node->HasResult()) {
    vStream << ",\nresults=";
    visit(node->GetResult());
  }
  vStream << "\nstatements=[\n";
  const ASTStatementList *statements = &node->GetStatements();
  BaseQASM3Visitor::visit(statements);
  vStream << "])\n";
}

void PrintQASM3Visitor::visit(const ASTGateDeclarationNode *node) {
  vStream << "GateDeclarationNode(";
  const ASTGateNode *gateNode = node->GetGateNode();
  const std::string &gateName = gateNode->GetName();
  vStream << "name=" << gateName << ", ";
  visit(gateNode);
  vStream << ")\n";
}

void PrintQASM3Visitor::visit(const ASTGenericGateOpNode *node) {
  vStream << "GateGenericNode(";
  const ASTGateNode *gateNode = node->GetGateNode();
  const std::string &gateName = gateNode->GetName();
  vStream << "name=" << gateName << ", ";
  visit(gateNode);
  vStream << ")\n";
}

void PrintQASM3Visitor::visit(const ASTGateNode *node) {
  const size_t numParams = node->ParamsSize();
  const size_t numQubits = node->QubitsSize();

  vStream << "params=[";
  for (size_t i = 0; i < numParams; i++) {
    if (i >= 1)
      vStream << ", ";
    visit(node->GetParam(i));
  }

  vStream << "], qubits=[";
  for (size_t i = 0; i < numQubits; i++) {
    if (i >= 1)
      vStream << ", ";
    visit(node->GetQubit(i));
  }
  vStream << "], qcparams=[";

  for (size_t i = 0; i < node->GetNumQCParams(); i++) {
    if (i >= 1)
      vStream << ", ";

    auto *paramId = node->GetQCParams()[i]->GetIdentifier();
    assert(paramId);
    vStream << paramId->GetName();
  }

  vStream << "]";

  if (node->HasOpList()) {
    const ASTGateQOpList &opList = node->GetOpList();
    vStream << ",\nops=[\n";
    for (ASTGateQOpNode *i : opList) {
      BaseQASM3Visitor::visit(i);
      vStream << ",\n";
    }
    vStream << "]\n";
  }
}

void PrintQASM3Visitor::visit(const ASTHGateOpNode *node) {
  vStream << "HGateOpNode(";
  const ASTGateNode *gateNode = node->GetGateNode();
  visit(gateNode);
  vStream << ")\n";
}

void PrintQASM3Visitor::visit(const ASTUGateOpNode *node) {
  vStream << "UGateOpNode(";
  const ASTGateNode *gateNode = node->GetGateNode();
  visit(gateNode);
  vStream << ")\n";
}

void PrintQASM3Visitor::visit(const ASTCXGateOpNode *node) {
  vStream << "CXGateOpNode(";
  const ASTGateNode *gateNode = node->GetGateNode();
  visit(gateNode);
  vStream << ")\n";
}

void PrintQASM3Visitor::visit(const ASTResetNode *node) {
  const ASTIdentifierNode *idNode = node->GetTarget();
  vStream << "ResetNode(";
  visit(idNode);
  vStream << ")\n";
}

void PrintQASM3Visitor::visit(const ASTMeasureNode *node) {
  const ASTQubitContainerNode *qubitNode = node->GetTarget();
  vStream << "MeasureNode(qubits=[";
  visit(qubitNode);
  vStream << "], result=";
  if (const ASTCBitNode *bits = node->GetResult())
    visit(bits);
  // include cbit index if is specified
  for (unsigned i = 0; i < node->GetResultSize(); i++)
    vStream << "[index=" << node->GetResultIndex(i) << "]";

  vStream << ")\n";
}

void PrintQASM3Visitor::visit(const ASTDelayStatementNode *node) {
  const ASTDelayNode *delayNode = node->GetDelay();
  vStream << "DelayStatementNode(";
  visit(delayNode);
  vStream << ")\n";
}

void PrintQASM3Visitor::visit(const ASTDelayNode *node) {
  vStream << "DelayNode(";
  switch (ASTType delayType = node->GetDelayType()) {
  case ASTTypeDuration: {
    vStream << "duration=";
    const ASTDurationNode *durationNode = node->GetDurationNode();
    vStream << durationNode->GetName();
    vStream << ", qubit=";
    const ASTIdentifierNode *delayIdNode = node->GetDelayQubitIdentifier();
    visit(delayIdNode);
    break;
  }
  case ASTTypeQubit: {
    vStream << "duration=";
    vStream << node->GetDuration();
    vStream << PrintLengthUnit(node->GetLengthUnit());
    vStream << ", qubit=";
    const ASTIdentifierList &delayIdNode = node->GetOperandList();
    for (ASTIdentifierNode *i : delayIdNode) {
      visit(i);
      vStream << ", ";
    }
    break;
  }
  case ASTTypeStretch: {
    vStream << "stretch=";
    const ASTStretchNode *stretchNode = node->GetStretchNode();
    visit(stretchNode);
    vStream << ", qubit=";
    const ASTIdentifierList &delayIdNode = node->GetOperandList();
    for (ASTIdentifierNode *i : delayIdNode) {
      visit(i);
      vStream << ", ";
    }
    break;
  }
  default: {
    std::ostringstream oss;
    oss << "Cannot process " << PrintTypeEnum(delayType) << " delay node.\n";
    throw std::runtime_error(oss.str());
  }
  }
  vStream << ")";
}

void PrintQASM3Visitor::visit(const ASTBarrierNode *node) {
  const ASTIdentifierList &idList = node->GetOperandList();
  vStream << "BarrierNode(ids=[\n";
  for (ASTIdentifierNode *i : idList) {
    visit(i);
    vStream << ",\n";
  }
  vStream << "])\n";
}

void PrintQASM3Visitor::visit(const ASTDeclarationNode *node) {
  ASTType declType = node->GetASTType();
  vStream << "DeclarationNode(type=" << PrintTypeEnum(declType) << ", ";
  // if it's a function, process it directly
  if (const auto *funcDecl =
          dynamic_cast<const ASTFunctionDeclarationNode *>(node)) {
    visit(funcDecl);
    return;
  }
  // otherwise, lookup node in sym table if it exists
  const ASTIdentifierNode *idNode = node->GetIdentifier();
  ASTSymbolTableEntry *symTableEntry =
      ASTSymbolTable::Instance().Lookup(idNode);
  if (symTableEntry)
    BaseQASM3Visitor::visit(symTableEntry);
  // finally resort to printing the identifier.
  else
    visit(idNode);
  if (node->GetModifierType() == QASM::ASTTypeInputModifier)
    vStream << ", inputVariable";
  else if (node->GetModifierType() == QASM::ASTTypeOutputModifier)
    vStream << ", outputVariable";
  vStream << ")\n";
}

void PrintQASM3Visitor::visit(const ASTKernelDeclarationNode *node) {
  visit(node->GetKernel());
  return;
}

void PrintQASM3Visitor::visit(const ASTQubitContainerNode *node) {
  vStream << "QubitContainerNode(";
  const unsigned numQubits = node->Size();
  for (unsigned i = 0; i < numQubits; ++i) {
    if (i >= 1)
      vStream << ", ";
    visit(node->GetQubit(i));
  }
  vStream << ")";
}

void PrintQASM3Visitor::visit(const ASTQubitNode *node) {
  const std::string &name = node->GetName();
  const unsigned bits = node->GetIdentifier()->GetBits();
  vStream << "QubitNode(name=" << name << ", bits=" << bits << ")";
}

void PrintQASM3Visitor::visit(const ASTCBitNode *node) {
  auto *identifier = node->GetIdentifier();

  if (identifier->GetASTType() == ASTTypeIdentifierRef) {
    const auto *refnode =
        dynamic_cast<const ASTIdentifierRefNode *>(identifier);
    assert(refnode && "ASTIdentifierNode of ASTTYpe ASTTypeIdentifierRef "
                      "should also be an ASTIdentifierRefNode");
    identifier = refnode->GetIdentifier();
  }

  const std::string &name = identifier->GetName();
  const unsigned bits = identifier->GetBits();
  std::string value = node->AsString();

  if (bits < value.size())
    value = value.substr(value.size() - bits);
  vStream << "CBitNode(name=" << name << ", bits=" << bits;
  if (node->IsSet(0))
    vStream << ", value=" << value;
  if (const auto *nodeGateOp =
          dynamic_cast<const ASTMeasureNode *>(node->GetGateQOp())) {
    vStream << ", ";
    visit(nodeGateOp);
  }
  vStream << ")";
}

void PrintQASM3Visitor::visit(const ASTDurationNode *node) {
  const uint64_t duration = node->GetDuration();
  const LengthUnit unit = node->GetLengthUnit();
  const std::string &name = node->GetName();
  vStream << "DurationNode(duration=" << duration
          << ", unit=" << PrintLengthUnit(unit) << ", name=" << name << ")";
}

void PrintQASM3Visitor::visit(const ASTStretchStatementNode *node) {
  const ASTStretchNode *stretchNode = node->GetStretch();
  vStream << "StretchStatementNode(";
  visit(stretchNode);
  vStream << ")\n";
}

void PrintQASM3Visitor::visit(const ASTStretchNode *node) {
  vStream << "StretchNode(name=" << node->GetName() << ")";
}

void PrintQASM3Visitor::visit(const ASTIdentifierRefNode *node) {
  vStream << "IdentifierRefNode(name=" << node->GetName() << ", ";
  visit(node->GetIdentifier());
  if (node->IsIndexed())
    vStream << ", index=" << node->GetIndex();
  vStream << ")";
}

void PrintQASM3Visitor::visit(const ASTIdentifierNode *node) {
  const unsigned bits = node->GetBits();
  const std::string &name = node->GetName();
  vStream << "IdentifierNode(name=" << name << ", bits=" << bits << ")";
}

void PrintQASM3Visitor::visit(const ASTBinaryOpNode *node) {
  ASTOpType opType = node->GetOpType();
  const ASTExpressionNode *left = node->GetLeft();
  const ASTExpressionNode *right = node->GetRight();
  vStream << "BinaryOpNode(type=" << PrintOpTypeEnum(opType) << ", left=";
  BaseQASM3Visitor::visit(left);
  vStream << ", right=";
  BaseQASM3Visitor::visit(right);
  vStream << ")\n";
}

void PrintQASM3Visitor::visit(const ASTUnaryOpNode *node) {
  ASTOpType opType = node->GetOpType();
  const auto *operand = node->GetExpression();

  vStream << "UnaryOpNode(type=" << PrintOpTypeEnum(opType) << ", operand=";
  BaseQASM3Visitor::visit(operand);
  vStream << ")";
}

void PrintQASM3Visitor::visit(const ASTIntNode *node) {
  const unsigned bits = node->GetBits();
  const bool isSigned = node->IsSigned();
  vStream << "IntNode(signed=";

  if (isSigned) {
    vStream << "true"
            << ", value=";
    const int32_t val = node->GetSignedValue();
    vStream << val;
  } else {
    vStream << "false"
            << ", value=";
    const uint32_t val = node->GetUnsignedValue();
    vStream << val;
  }
  vStream << ", bits=" << bits;

  vStream << ")";
}

void PrintQASM3Visitor::visit(const ASTMPIntegerNode *node) {
  vStream << "MPIntegerNode(name=" << node->GetName()
          << ", value=" << node->GetValue() << ", bits=" << node->GetBits()
          << ", signed=" << node->IsSigned() << ")";
}

void PrintQASM3Visitor::visit(const ASTFloatNode *node) {
  vStream << "FloatNode(name=" << node->GetName()
          << ", value=" << node->GetValue() << ", bits=" << node->GetBits()
          << ")";
}

void PrintQASM3Visitor::visit(const ASTMPDecimalNode *node) {
  const unsigned bits = node->GetIdentifier()->GetBits();
  std::string val = node->GetValue();
  vStream << "MPDecimalNode(name=" << node->GetName();
  if (!node->IsNan())
    vStream << ", value=" << val;
  vStream << ", bits=" << bits << ")";
}

void PrintQASM3Visitor::visit(const ASTMPComplexNode *node) {
  const std::string &name = node->GetName();
  const std::string val = node->GetValue();
  const unsigned bits = node->GetIdentifier()->GetBits();
  int position = val.find(' ');
  std::string real = val.substr(1, position - 1);
  std::string imag = val.substr(position + 1, val.length() - position - 2);

  vStream << "MPComplexNode(name=" << name;
  if (!node->IsNan()) {
    vStream << ", value=";
    vStream << real << " + " << imag << " im";
  }
  vStream << ", bits=" << bits << ")";
}

void PrintQASM3Visitor::visit(const ASTAngleNode *node) {
  std::string val = node->GetValue();
  const unsigned bits = node->GetBits();
  if (node->IsNan())
    val = "0.0";
  vStream << "AngleNode(value=" << val << ", bits=" << bits << ")";
}

void PrintQASM3Visitor::visit(const ASTBoolNode *node) {
  vStream << "BoolNode(name=" << node->GetName() << ", ";
  if (node->GetValue())
    vStream << "true)";
  else
    vStream << "false)";
}

void PrintQASM3Visitor::visit(const ASTCastExpressionNode *node) {
  vStream << "CastNode(from=" << PrintTypeEnum(node->GetCastFrom());
  vStream << ", to=" << PrintTypeEnum(node->GetCastTo()) << ", expression=";
  // dispatch into cast operand
  BaseQASM3Visitor::visit(node);
}

void PrintQASM3Visitor::visit(const ASTKernelNode *node) {
  vStream << "ExternNode(name=" << node->GetName();
  vStream << ", parameters=[";
  BaseQASM3Visitor::visit(&node->GetParameters());
  vStream << "], returns=";
  visit(node->GetResult());
  vStream << ")\n";
}

void PrintQASM3Visitor::visit(const ASTDeclarationList *list) {
  if (!list) {
    vStream << "void";
    return;
  }
  for (ASTDeclarationNode *declNode : *list) {
    visit(declNode);
    if (!(declNode == list->back()))
      vStream << ", ";
  }
}

void PrintQASM3Visitor::visit(const ASTFunctionCallNode *node) {
  vStream << "FunctionCallNode(type=";
  switch (node->GetFunctionCallType()) {
  case ASTTypeFunctionCallExpression:
    vStream << "ASTTypeFunctionCallExpression"
            << ", expressions=";
    BaseQASM3Visitor::visit(&node->GetExpressionList());
    vStream << ", quantumIdentifiers=";
    BaseQASM3Visitor::visit(&node->GetQuantumIdentifierList());
    break;
  case ASTTypeKernelCallExpression:
    vStream << "ASTTypeFunctionCallExpression"
            << ", expressions=";
    BaseQASM3Visitor::visit(&node->GetExpressionList());
    vStream << ", kernelDefinition=";
    BaseQASM3Visitor::visit(node->GetKernelDefinition());
    break;
  default:
    llvm::errs() << "Unhandled call type "
                 << PrintTypeEnum(node->GetFunctionCallType()) << "\n";
    llvm_unreachable("unhandled call type in ASTFunctionCallNode");
  }
  vStream << ")\n";
}

void PrintQASM3Visitor::visit(const QASM::ASTVoidNode *) { vStream << "void"; }

void PrintQASM3Visitor::visit(const QASM::ASTOperatorNode *node) {
  vStream << "OperatorNode(";
  vStream << "op=" << PrintOpTypeOperator(node->GetOpType()) << ", ";
  if (const auto *id = node->GetTargetIdentifier()) {
    vStream << "target-identifier=";
    if (id->IsReference())
      BaseQASM3Visitor::dispatchVisit<ASTIdentifierRefNode>(id);
    else
      BaseQASM3Visitor::dispatchVisit<ASTIdentifierNode>(id);
  } else {
    vStream << "target-expression=";
    BaseQASM3Visitor::visit(node->GetTargetExpression());
  }
  vStream << ")\n";
}

} // namespace qssc::frontend::openqasm3
