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

#include "Dialect/QCS/IR/QCSOps.h"
#include "Dialect/QUIR/IR/QUIRAttributes.h"
#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIRTraits.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <set>
#include <string>
#include <sys/types.h>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::quir;

namespace {
uint lookupQubitIdHandleError_(const Value &val) {
  auto id = lookupQubitId(val);
  if (!id.has_value()) {
    auto &diagEngine = val.getContext()->getDiagEngine();
    diagEngine.emit(val.getLoc(), mlir::DiagnosticSeverity::Error)
        << "Qubit does not have a valid ID.";
    val.getDefiningOp()->emitError() << "Qubit does not have a valid ID.";
    assert(false && "Qubit does not have a valid id.");
  }
  return id.value();
} // lookupQubitIdHandleError_
} // anonymous namespace

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
  opQubits.insert(lookupQubitIdHandleError_(getControl()));
  opQubits.insert(lookupQubitIdHandleError_(getTarget()));
  return opQubits;
}

//===----------------------------------------------------------------------===//
// Builtin_UOp
//===----------------------------------------------------------------------===//

std::set<uint32_t> Builtin_UOp::getOperatedQubits() {
  std::set<uint32_t> opQubits;
  opQubits.insert(lookupQubitIdHandleError_(getTarget()));
  return opQubits;
}

//===----------------------------------------------------------------------===//
// CallGateOp
//===----------------------------------------------------------------------===//

std::set<uint32_t> CallGateOp::getOperatedQubits() {
  return getQubitIds<CallGateOp>(*this);
}

auto CallGateOp::getCalleeType() -> FunctionType {
  return FunctionType::get(getContext(), getOperandTypes(), {});
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
  if (value.isa<AngleAttr>() && !type.isa<quir::AngleType>())
    return false;
  // Currently only supports angle attributes to create angle constants
  return value.isa<quir::AngleAttr>();
}

// Returns the float value from the attribute on this constant op
APFloat quir::ConstantOp::getAngleValueFromConstant() {
  auto attr = getValue().dyn_cast<quir::AngleAttr>();
  assert(attr && "Trying to get the angle attribute of a non-angle constantOp");
  return attr.getValue();
}

LogicalResult CallDefcalMeasureOp::verify() {
  bool qubitFound = false;
  for (auto arg : (*this)->getOperands()) {
    if (qubitFound) {
      if (arg.getType().isa<QubitType>())
        return emitOpError("requires exactly one qubit argument");
    } else {
      if (arg.getType().isa<QubitType>())
        qubitFound = true;
    }
  }
  if (qubitFound)
    return success();

  return emitOpError("requires exactly one qubit");
}

mlir::OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  assert(adaptor.getOperands().empty() && "constant has no operands");
  return getValue();
}

/// Materialize a constant, can be any buildable type, used by canonicalization
ConstantOp ConstantOp::materialize(OpBuilder &builder, Attribute value,
                                   Type type, Location loc) {
  return builder.create<quir::ConstantOp>(loc, cast<TypedAttr>(value));
}

/// Materialize a constant, can be any buildable type, used by canonicalization
Operation *QUIRDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                            Type type, Location loc) {
  return builder.create<quir::ConstantOp>(loc, type, cast<TypedAttr>(value));
}

//===----------------------------------------------------------------------===//
// CallDefcalMeasureOp
//===----------------------------------------------------------------------===//

auto CallDefcalMeasureOp::getCalleeType() -> FunctionType {
  return FunctionType::get(getContext(), getOperandTypes(),
                           getResult().getType());
}

//===----------------------------------------------------------------------===//
// CallDefCalGateOp
//===----------------------------------------------------------------------===//

auto CallDefCalGateOp::getCalleeType() -> FunctionType {
  return FunctionType::get(getContext(), getOperandTypes(), {});
}

//===----------------------------------------------------------------------===//
// CallSubroutineOp
//===----------------------------------------------------------------------===//

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
  auto circuitType = circuit.getFunctionType();
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

mlir::ParseResult CircuitOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  auto buildFuncType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false,
      getFunctionTypeAttrName(result.name), buildFuncType,
      getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
}

void CircuitOp::print(mlir::OpAsmPrinter &printer) {
  function_interface_impl::printFunctionOp(
      printer, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

namespace {
/// Verify the argument list and entry block are in agreement.
LogicalResult verifyArgumentAndEntry_(CircuitOp op) {
  auto fnInputTypes = op.getFunctionType().getInputs();
  Block &entryBlock = op.front();
  for (unsigned i = 0; i != entryBlock.getNumArguments(); ++i)
    if (fnInputTypes[i] != entryBlock.getArgument(i).getType())
      return op.emitOpError("type of entry block argument #")
             << i << '(' << entryBlock.getArgument(i).getType()
             << ") must match the type of the corresponding argument in "
             << "function signature(" << fnInputTypes[i] << ')';
  return success();
}

/// Verify that no classical values are created/used in the circuit outside of
/// values that originate as argument values or the result of a measurement.
LogicalResult verifyClassical_(CircuitOp op) {
  mlir::Operation *classicalOp = nullptr;
  WalkResult const result = op->walk([&](Operation *subOp) {
    if (isa<mlir::arith::ConstantOp>(subOp) || isa<quir::ConstantOp>(subOp) ||
        isa<qcs::ParameterLoadOp>(subOp) || isa<CallCircuitOp>(subOp) ||
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
} // anonymous namespace

LogicalResult CircuitOp::verify() {
  // If external will be linked in later and nothing to do
  if (isExternal())
    return success();

  if (failed(verifyArgumentAndEntry_(*this)))
    return mlir::failure();

  if (failed(verifyClassical_(*this)))
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
  SmallVector<NamedAttribute, 8> const attrRef(attrs);
  return create(location, name, type, attrRef);
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
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty())
    return;
  assert(type.getNumInputs() == argAttrs.size());
  function_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, /*resultAttrs=*/std::nullopt,
      getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
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
  FunctionType newType = getFunctionType();

  // If the function contains a body, then its possible arguments
  // may be deleted in the mapper. Verify this so they aren't
  // added to the input type vector.
  bool const isExternalCircuit = isExternal();
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

LogicalResult mlir::quir::ReturnOp::verify() {
  auto circuit = (*this)->getParentOfType<CircuitOp>();

  FunctionType const circuitType = circuit.getFunctionType();

  auto numResults = circuitType.getNumResults();
  // Verify number of operands match type signature
  if (numResults != getOperands().size()) {
    return emitError()
        .append("expected ", numResults, " result operands")
        .attachNote(circuit.getLoc())
        .append("return type declared here");
  }

  int i = 0;
  for (const auto [type, operand] :
       llvm::zip(circuitType.getResults(), getOperands())) {
    auto opType = operand.getType();
    if (type != opType) {
      return emitOpError() << "unexpected type `" << opType << "' for operand #"
                           << i;
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
  OpAsmParser::UnresolvedOperand flag;
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
    OptionalParseResult const integerParseResult =
        parser.parseOptionalInteger(caseVal);
    if (!integerParseResult.has_value() || integerParseResult.value())
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
  bool printBlockTerminators = false;

  printer << " " << getFlag();
  if (!getResultTypes().empty()) {
    printer << " -> (" << getResultTypes().getTypes() << ")";
    // Print yield explicitly if the op defines values.
    printBlockTerminators = true;
  }
  printer.printRegion(getDefaultRegion(),
                      /*printEntryBlockOperands=*/false,
                      /*printBlockTerminators=*/printBlockTerminators);

  printer << "[";

  uint64_t id = 0;

  for (auto &region : getCaseRegions())
    if (!(region.empty())) {
      printer.printAttributeWithoutType(
          getCaseValuesAttr().getValues<Attribute>()[id]);
      id += 1;
      printer << " : ";
      printer.printRegion(region,
                          /*printEntryBlockOperands=*/false,
                          /*printBlockTerminators=*/printBlockTerminators);
    }
  printer << "]";
  printer.printOptionalAttrDict((*this)->getAttrs().drop_front(1));
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
  for (auto *region = getCaseRegions().begin();
       region != getCaseRegions().end(); region++)
    regions.push_back(region);
}

LogicalResult quir::SwitchOp::verify() {
  if ((!getCaseValues() && !getCaseRegions().empty()) ||
      (getCaseValues() &&
       getCaseValues().size() != static_cast<int64_t>(getCaseRegions().size())))
    return emitOpError("expects same number of case values and case regions");

  return success();
}

//===----------------------------------------------------------------------===//
// end SwitchOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult quir::YieldOp::verify() {
  auto *parentOp = (*this)->getParentOp();
  auto results = parentOp->getResults();
  auto operands = getOperands();

  if (!isa<quir::SwitchOp>(parentOp))
    return emitOpError() << "only terminates quir.switch regions";
  if (parentOp->getNumResults() != getNumOperands())
    return emitOpError() << "parent of yield must have same number of "
                            "results as the yield operands";
  for (auto it : llvm::zip(results, operands))
    if (std::get<0>(it).getType() != std::get<1>(it).getType())
      return emitOpError() << "types mismatch between yield op and its parent";

  return success();
}

//===----------------------------------------------------------------------===//
// end YieldOp
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Table generated op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/QUIR/IR/QUIR.cpp.inc"

#define GET_ENUM_CLASSES
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/QUIR/IR/QUIREnums.cpp.inc"
