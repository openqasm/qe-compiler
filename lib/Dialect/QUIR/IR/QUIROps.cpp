//===- QUIROps.cpp - QUIR dialect ops ---------------------------*- C++ -*-===//
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

#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/OQ3/IR/OQ3Ops.h"
#include "Dialect/QCS/IR/QCSOps.h"
#include "Dialect/QUIR/IR/QUIRAttributes.h"
#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;
using namespace mlir::quir;

static uint lookupQubitIdHandleError_(const Value &val) {
  auto id = lookupQubitId(val);
  if (!id.has_value()) {
    auto &diagEngine = val.getContext()->getDiagEngine();
    diagEngine.emit(val.getLoc(), mlir::DiagnosticSeverity::Error)
        << "Qubit does not have a valid ID.";
    val.getDefiningOp()->emitError() << "Qubit does not have a valid ID.";
  }
  return id.getValue();
} // lookupQubitIdHandleError_

template <class Op>
std::set<uint32_t> getQubitIds(Op &op) {
  std::set<uint32_t> opQubits;
  std::vector<Value> vals;
  qubitCallOperands<Op>(op, vals);
  for (auto qubit : vals) {
    auto id = lookupQubitIdHandleError_(qubit);
    opQubits.insert(id);
  }
  return opQubits;
} // getQubitIds

//===----------------------------------------------------------------------===//
// BuiltinCXOp
//===----------------------------------------------------------------------===//

std::set<uint32_t> BuiltinCXOp::getOperatedQubits() {
  std::set<uint32_t> opQubits;
  opQubits.insert(lookupQubitIdHandleError_(control()));
  opQubits.insert(lookupQubitIdHandleError_(target()));
  return opQubits;
}

//===----------------------------------------------------------------------===//
// Builtin_UOp
//===----------------------------------------------------------------------===//

std::set<uint32_t> Builtin_UOp::getOperatedQubits() {
  std::set<uint32_t> opQubits;
  opQubits.insert(lookupQubitIdHandleError_(target()));
  return opQubits;
}

//===----------------------------------------------------------------------===//
// CallGateOp
//===----------------------------------------------------------------------===//

std::set<uint32_t> CallGateOp::getOperatedQubits() {
  return getQubitIds<CallGateOp>(*this);
}

//===----------------------------------------------------------------------===//
// MeasureOp
//===----------------------------------------------------------------------===//

std::set<uint32_t> MeasureOp::getOperatedQubits() {
  return getQubitIds<MeasureOp>(*this);
}

//===----------------------------------------------------------------------===//
// ResetOp
//===----------------------------------------------------------------------===//

std::set<uint32_t> ResetQubitOp::getOperatedQubits() {
  return getQubitIds<ResetQubitOp>(*this);
}

//===----------------------------------------------------------------------===//
// SynchronizeOp
//
// TODO: Move to `System` dialect once "lower qubits to channels/ports."
//===----------------------------------------------------------------------===//

std::set<uint32_t> qcs::SynchronizeOp::getOperatedQubits() {
  return getQubitIds<qcs::SynchronizeOp>(*this);
}

//===----------------------------------------------------------------------===//
// DelayOp
//===----------------------------------------------------------------------===//

std::set<uint32_t> DelayOp::getOperatedQubits() {
  return getQubitIds<DelayOp>(*this);
}

//===----------------------------------------------------------------------===//
// DelayCyclesOp
//
// TODO: Move to `System` dialect once "lower qubits to channels/ports."
//===----------------------------------------------------------------------===//

std::set<uint32_t> qcs::DelayCyclesOp::getOperatedQubits() {
  return getQubitIds<qcs::DelayCyclesOp>(*this);
}

//===----------------------------------------------------------------------===//
// BarrierOp
//===----------------------------------------------------------------------===//

std::set<uint32_t> BarrierOp::getOperatedQubits() {
  return getQubitIds<BarrierOp>(*this);
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult quir::ConstantOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.empty() && "constant has no operands");
  return value();
}

void quir::ConstantOp::getAsmResultNames(
    function_ref<void(Value, StringRef)> setNameFn) {
  auto type = getType();
  if (type.isa<AngleType>())
    setNameFn(getResult(), "angle");
  else if (type.isa<DurationType>())
    setNameFn(getResult(), "dur");
  else
    setNameFn(getResult(), "qcst");
}

/// Returns true if a constant operation can be built with the given value and
/// result type.
bool quir::ConstantOp::isBuildableWith(Attribute value, Type type) {
  // The value's type must be the same as the provided type.
  if (value.getType().isa<AngleType>() && !type.isa<quir::AngleType>())
    return false;
  // Currently only supports angle attributes to create angle constants
  return value.isa<quir::AngleAttr>();
}

// Returns the float value from the attribute on this constant op
APFloat quir::ConstantOp::getAngleValueFromConstant() {
  auto attr = value().dyn_cast<quir::AngleAttr>();
  assert(attr && "Trying to get the angle attribute of a non-angle constantOp");
  return attr.getValue();
}

/// Return the callee of the call_gate operation, this is
/// required by the call interface.
auto CallGateOp::getCallableForCallee() -> CallInterfaceCallable {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
auto CallGateOp::getArgOperands() -> mlir::OperandRange {
  return getOperands();
}

auto CallGateOp::getCalleeType() -> FunctionType {
  return FunctionType::get(getContext(), getOperandTypes(), TypeRange{});
}

// CallDefCalGateOp, CallDefcalMeasureOp
auto CallDefCalGateOp::getCallableForCallee() -> CallInterfaceCallable {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}
auto CallDefcalMeasureOp::getCallableForCallee() -> CallInterfaceCallable {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

auto CallDefCalGateOp::getArgOperands() -> mlir::OperandRange {
  return getOperands();
}
auto CallDefcalMeasureOp::getArgOperands() -> mlir::OperandRange {
  return getOperands();
}

auto CallDefCalGateOp::getCalleeType() -> FunctionType {
  return FunctionType::get(getContext(), getOperandTypes(), TypeRange{});
}

auto CallDefcalMeasureOp::getCalleeType() -> FunctionType {
  return FunctionType::get(getContext(), getOperandTypes(),
                           getOperation()->getResultTypes());
}

LogicalResult CallDefcalMeasureOp::verify() {
  bool qubitFound = false;
  for (auto arg : (*this)->getOperands()) {
      if(qubitFound) {
          if(arg.getType().isa<QubitType>()) {
              return emitOpError("requires exactly one qubit argument");
          }
      } else {
          if(arg.getType().isa<QubitType>()) {
              qubitFound = true;
          }
      }
  }
  if (qubitFound) {
      return success();
  } else {
      return emitOpError("requires exactly one qubit");
  }
}

/// Return the callee of the call_gate operation, this is
/// required by the call interface.
auto CallSubroutineOp::getCallableForCallee() -> CallInterfaceCallable {
  return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
auto CallSubroutineOp::getArgOperands() -> mlir::OperandRange {
  return getOperands();
}

auto CallSubroutineOp::getCalleeType() -> FunctionType {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

//===----------------------------------------------------------------------===//
// CallCircuitOp
//===----------------------------------------------------------------------===//

auto CallCircuitOp::getCalleeType() -> FunctionType {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

LogicalResult
CallCircuitOp::verifySymbolUses(SymbolTableCollection &symbolTable) {

  auto circuitAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!circuitAttr)
    return emitOpError("Requires a 'callee' symbol reference attribute");

  auto circuit =
      symbolTable.lookupNearestSymbolFrom<CircuitOp>(*this, circuitAttr);
  if (!circuit)
    return emitOpError() << "'" << circuitAttr.getValue()
                         << "' does not reference a valid circuit";

  // Verify the types match
  auto circuitType = circuit.getType();
  if (circuitType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for the callee circuit");

  for (unsigned i = 0; i != circuitType.getNumInputs(); ++i) {
    if (getOperand(i).getType() != circuitType.getInput(i)) {
      auto diag = emitOpError("operand type mismatch at index ") << i;
      diag.attachNote() << "op input types: " << getOperandTypes();
      diag.attachNote() << "function operand types: "
                        << circuitType.getInputs();
      return diag;
    }
  }

  if (circuitType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for the callee circuit");

  for (unsigned i = 0; i != circuitType.getNumResults(); ++i) {
    if (getResult(i).getType() != circuitType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "op result types: " << getResultTypes();
      diag.attachNote() << "function result types: "
                        << circuitType.getResults();
      return diag;
    }
  }

  return success();
}

std::set<uint32_t> CallCircuitOp::getOperatedQubits() {
  return getQubitIds<CallCircuitOp>(*this);
}

//===----------------------------------------------------------------------===//
//
// end CallCircuitOp
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//
// This code section was derived and modified from the LLVM project
// Consequently it is licensed as Apache-2.0 WITH LLVM-exception.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CircuitOp
//
// This code section was derived and modified from the LLVM project FuncOp
// Consequently it is licensed as Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

mlir::ParseResult SequenceOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  auto buildCircuitType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };
  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false, buildCircuitType);
}

void SequenceOp::print(mlir::OpAsmPrinter &printer) {
  FunctionType fnType = getType();
  function_interface_impl::printFunctionOp(
      printer, *this, fnType.getInputs(), /*isVariadic=*/false, fnType.getResults());
}

/// Verify the argument list and entry block are in agreement.
static LogicalResult verifyArgumentAndEntry_(CircuitOp op) {
  auto fnInputTypes = op.getType().getInputs();
  Block &entryBlock = op.front();
  for (unsigned i = 0; i != entryBlock.getNumArguments(); ++i) {
    if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
      return op.emitOpError("type of entry block argument #")
             << i << '(' << entryBlock.getArgument(i).getType()
             << ") must match the type of the corresponding argument in "
             << "function signature(" << fnInputTypes[i] << ')';
  }
  return success();
}

/// Verify that no classical values are created/used in the circuit outside of
/// values that originate as argument values or the result of a measurement.
static LogicalResult verifyClassical_(CircuitOp op) {
  mlir::Operation *classicalOp = nullptr;
  WalkResult result = op->walk([&](Operation *subOp) {
    if (isa<mlir::arith::ConstantOp>(subOp) || isa<mlir::arith::ConstantOp>(subOp) ||
        isa<quir::ConstantOp>(subOp) || isa<CallCircuitOp>(subOp) ||
        isa<quir::ReturnOp>(subOp) || isa<CircuitOp>(subOp) ||
        subOp->hasTrait<mlir::quir::UnitaryOp>() ||
        subOp->hasTrait<mlir::quir::CPTPOp>())
      return WalkResult::advance();
    classicalOp = subOp;
    return WalkResult::interrupt();
  });

  if (result.wasInterrupted())
    return classicalOp->emitOpError()
           << "is classical and should not be inside a circuit.";
  return success();
}

static LogicalResult verify(CircuitOp op) {
  // If external will be linked in later and nothing to do
  if (op.isExternal())
    return success();

  if (failed(verifyArgumentAndEntry_(op)))
    return mlir::failure();

  if (failed(verifyClassical_(op)))
    return mlir::failure();

  return success();
}

CircuitOp CircuitOp::create(Location location, StringRef name,
                            FunctionType type, ArrayRef<NamedAttribute> attrs) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  CircuitOp::build(builder, state, name, type, attrs);
  return cast<CircuitOp>(Operation::create(state));
}
CircuitOp CircuitOp::create(Location location, StringRef name,
                            FunctionType type,
                            Operation::dialect_attr_range attrs) {
  SmallVector<NamedAttribute, 8> attrRef(attrs);
  return create(location, name, type, llvm::makeArrayRef(attrRef));
}
CircuitOp CircuitOp::create(Location location, StringRef name,
                            FunctionType type, ArrayRef<NamedAttribute> attrs,
                            ArrayRef<DictionaryAttr> argAttrs) {
  CircuitOp circ = create(location, name, type, attrs);
  circ.setAllArgAttrs(argAttrs);
  return circ;
}

void CircuitOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                      FunctionType type, ArrayRef<NamedAttribute> attrs,
                      ArrayRef<DictionaryAttr> argAttrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getTypeAttrName(), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(builder, state, argAttrs,
                                                /*resultAttrs=*/std::nullopt);
}

/// Clone the internal blocks and attributes from this circuit to the
/// destination circuit.
void CircuitOp::cloneInto(CircuitOp dest, IRMapping &mapper) {
  // Add the attributes of this function to dest.
  llvm::MapVector<StringAttr, Attribute> newAttrMap;
  for (const auto &attr : dest->getAttrs())
    newAttrMap.insert({attr.getName(), attr.getValue()});
  for (const auto &attr : (*this)->getAttrs())
    newAttrMap.insert({attr.getName(), attr.getValue()});

  auto newAttrs = llvm::to_vector(llvm::map_range(
      newAttrMap, [](std::pair<StringAttr, Attribute> attrPair) {
        return NamedAttribute(attrPair.first, attrPair.second);
      }));
  dest->setAttrs(DictionaryAttr::get(getContext(), newAttrs));

  // Clone the body.
  getBody().cloneInto(&dest.getBody(), mapper);
}

/// Create a deep copy of this circuit and all of its block.
/// Remap any operands that use values outside of the function
/// Using the provider mapper. Replace references to
/// cloned sub-values with the corresponding copied value and
/// add to the mapper
CircuitOp CircuitOp::clone(IRMapping &mapper) {
  FunctionType newType = getType();

  // If the function contains a body, then its possible arguments
  // may be deleted in the mapper. Verify this so they aren't
  // added to the input type vector.
  bool isExternalCircuit = isExternal();
  if (!isExternalCircuit) {
    SmallVector<Type, 4> inputTypes;
    inputTypes.reserve(newType.getNumInputs());
    for (unsigned i = 0; i != getNumArguments(); ++i)
      if (!mapper.contains(getArgument(i)))
        inputTypes.push_back(newType.getInput(i));
    newType = FunctionType::get(getContext(), inputTypes, newType.getResults());
  }

  // Create the new circuit
  CircuitOp newCirc = cast<CircuitOp>(getOperation()->cloneWithoutRegions());
  newCirc.setType(newType);

  // Clone the current function into the new one and return.
  cloneInto(newCirc, mapper);
  return newCirc;
}

CircuitOp CircuitOp::clone() {
  IRMapping mapper;
  return clone(mapper);
}

//===----------------------------------------------------------------------===//
//
// end CircuitOp

//===----------------------------------------------------------------------===//
// ReturnOp
//
// This code section was derived and modified from the LLVM project's standard
// dialect ReturnOp.
// Consequently it is licensed as Apache-2.0 WITH LLVM-exception.
//
//===----------------------------------------------------------------------===//

static LogicalResult verify(mlir::quir::ReturnOp op) {
  auto circuit = op->getParentOfType<CircuitOp>();

  FunctionType circuitType = circuit.getType();

  auto numResults = circuitType.getNumResults();
  // Verify number of operands match type signature
  if (numResults != op.getOperands().size()) {
    return op.emitError()
        .append("expected ", numResults, " result operands")
        .attachNote(circuit.getLoc())
        .append("return type declared here");
  }

  int i = 0;
  for (const auto [type, operand] :
       llvm::zip(circuitType.getResults(), op.getOperands())) {
    auto opType = operand.getType();
    if (type != opType) {
      return op.emitOpError()
             << "unexpected type `" << opType << "' for operand #" << i;
    }
    i++;
  }
  return success();
}

//===----------------------------------------------------------------------===//
//
// end ReturnOp
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// end Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// SwitchOp
//===----------------------------------------------------------------------===//
mlir::ParseResult SwitchOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  Builder &builder = parser.getBuilder();
  OpAsmParser::OperandType flag;
  Region *defaultRegion;
  SmallVector<uint32_t> caseValues;
  ElementsAttr caseValuesAttr;

  if (parser.parseOperand(flag) ||
      parser.resolveOperand(flag, builder.getI32Type(), result.operands))
    return failure();

  // Parse optional results type list.
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();

  // Parse the default region.
  defaultRegion = result.addRegion();
  if (parser.parseRegion(*defaultRegion, /*arguments=*/{}, /*argTypes=*/{}))
    return failure();
  quir::SwitchOp::ensureTerminator(*defaultRegion, parser.getBuilder(),
                                   result.location);

  if (parser.parseLSquare())
    return failure();

  while (parser.parseOptionalRSquare()) {
    uint32_t caseVal;
    OptionalParseResult integerParseResult =
        parser.parseOptionalInteger(caseVal);
    if (!integerParseResult.has_value() || integerParseResult.getValue())
      // getValue() returns the success() or failure()
      return failure();
    caseValues.push_back(caseVal);

    Region *caseRegion = result.addRegion();
    if (parser.parseColon() ||
        parser.parseRegion(*caseRegion, /*arguments=*/{}, /*argTypes=*/{}))
      return failure();
    quir::SwitchOp::ensureTerminator(*caseRegion, parser.getBuilder(),
                                     result.location);
  }

  if (!caseValues.empty())
    caseValuesAttr = DenseIntElementsAttr::get(
        VectorType::get(static_cast<int64_t>(caseValues.size()),
                        builder.getIntegerType(32)),
        caseValues);
  if (caseValuesAttr)
    result.addAttribute("caseValues", caseValuesAttr);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void SwitchOp::print(mlir::OpAsmPrinter &printer) {
  print(printer, *this);
}


/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void quir::SwitchOp::getSuccessorRegions(
    std::optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  // regions branch back to parent operation
  if (index.has_value()) {
    regions.push_back(RegionSuccessor(getResults()));
    return;
  }

  // all regions are possible to be the successor
  regions.push_back(RegionSuccessor(&getDefaultRegion()));
  for (auto *region = caseRegions().begin(); region != caseRegions().end();
       region++)
    regions.push_back(region);
}

static void print(OpAsmPrinter &p, quir::SwitchOp op) {
  bool printBlockTerminators = false;

  p << " " << op.flag();
  if (!op.resultTypes().empty()) {
    p << " -> (" << op.getResultTypes() << ")";
    // Print yield explicitly if the op defines values.
    printBlockTerminators = true;
  }
  p.printRegion(op.getDefaultRegion(),
                /*printEntryBlockOperands=*/false,
                /*printBlockTerminators=*/printBlockTerminators);

  p << "[";

  uint64_t id = 0;

  for (auto *region = (op.caseRegions()).begin();
       region != (op.caseRegions()).end(); region++)
    if (!(region->empty())) {
      p.printAttributeWithoutType(
          op.caseValuesAttr().getValues<Attribute>()[id]);
      id += 1;
      p << " : ";
      p.printRegion(*region,
                    /*printEntryBlockOperands=*/false,
                    /*printBlockTerminators=*/printBlockTerminators);
    }
  p << "]";
  p.printOptionalAttrDict(op->getAttrs().drop_front(1));
}

static LogicalResult verify(quir::SwitchOp op) {
  if ((!op.caseValues() && !op.caseRegions().empty()) ||
      (op.caseValues() &&
       op.caseValues().size() != static_cast<int64_t>(op.caseRegions().size())))
    return op.emitOpError(
        "expects same number of case values and case regions");

  return success();
}

//===----------------------------------------------------------------------===//
// end SwitchOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

static LogicalResult verify(quir::YieldOp op) {
  auto *parentOp = op->getParentOp();
  auto results = parentOp->getResults();
  auto operands = op.getOperands();

  if (!isa<quir::SwitchOp>(parentOp))
    return op.emitOpError() << "only terminates quir.switch regions";
  if (parentOp->getNumResults() != op.getNumOperands())
    return op.emitOpError() << "parent of yield must have same number of "
                               "results as the yield operands";
  for (auto it : llvm::zip(results, operands)) {
    if (std::get<0>(it).getType() != std::get<1>(it).getType())
      return op.emitOpError()
             << "types mismatch between yield op and its parent";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// end YieldOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Table generated op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/QUIR/IR/QUIR.cpp.inc"

#define GET_ENUM_CLASSES
#include "Dialect/QUIR/IR/QUIREnums.cpp.inc"
