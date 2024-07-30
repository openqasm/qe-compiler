//===- PulseOps.cpp - Pulse dialect ops -------------------------*- C++ -*-===//
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

#include "Dialect/Pulse/IR/PulseOps.h"

#include "Dialect/Pulse/IR/PulseTraits.h"
#include "Dialect/QCS/IR/QCSOps.h"
#include "Dialect/QUIR/IR/QUIROps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace mlir::pulse {

//===----------------------------------------------------------------------===//
// Waveform Ops
//===----------------------------------------------------------------------===//

mlir::LogicalResult GaussianOp::verify() {
  auto durDeclOp = dyn_cast_or_null<mlir::arith::ConstantIntOp>(
      (*this).getDur().getDefiningOp());
  if (durDeclOp && durDeclOp.value() < 0)
    return emitOpError("duration must be >= 0.");
  return success();
}

mlir::LogicalResult GaussianSquareOp::verify() {
  auto durDeclOp = dyn_cast_or_null<mlir::arith::ConstantIntOp>(
      (*this).getDur().getDefiningOp());
  if (durDeclOp && durDeclOp.value() < 0)
    return emitOpError("duration must be >= 0.");
  return success();
}

mlir::LogicalResult DragOp::verify() {
  auto durDeclOp = dyn_cast_or_null<mlir::arith::ConstantIntOp>(
      (*this).getDur().getDefiningOp());
  if (durDeclOp && durDeclOp.value() < 0)
    return emitOpError("duration must be >= 0.");
  return success();
}

mlir::LogicalResult ConstOp::verify() {
  auto durDeclOp = dyn_cast_or_null<mlir::arith::ConstantIntOp>(
      (*this).getDur().getDefiningOp());
  if (durDeclOp && durDeclOp.value() < 0)
    return emitOpError("duration must be >= 0.");
  return success();
}

//===----------------------------------------------------------------------===//
// Waveform Ops
//===----------------------------------------------------------------------===//

mlir::LogicalResult SetFrequencyOp::verify() {
  if (!(*this).getTarget().isa<BlockArgument>())
    return emitOpError("Target is not a block argument; Target needs to be an "
                       "argument of pulse.sequence");
  return success();
}

mlir::LogicalResult ShiftFrequencyOp::verify() {
  if (!(*this).getTarget().isa<BlockArgument>())
    return emitOpError("Target is not a block argument; Target needs to be an "
                       "argument of pulse.sequence");
  return success();
}

mlir::LogicalResult SetPhaseOp::verify() {
  if (!(*this).getTarget().isa<BlockArgument>())
    return emitOpError("Target is not a block argument; Target needs to be an "
                       "argument of pulse.sequence");
  return success();
}

mlir::LogicalResult ShiftPhaseOp::verify() {
  if (!(*this).getTarget().isa<BlockArgument>())
    return emitOpError("Target is not a block argument; Target needs to be an "
                       "argument of pulse.sequence");
  return success();
}

mlir::LogicalResult SetAmplitudeOp::verify() {
  if (!(*this).getTarget().isa<BlockArgument>())
    return emitOpError("Target is not a block argument; Target needs to be an "
                       "argument of pulse.sequence");
  return success();
}

mlir::LogicalResult CaptureOp::verify() {
  if (!(*this).getTarget().isa<BlockArgument>())
    return emitOpError("Target is not a block argument; Target needs to be an "
                       "argument of pulse.sequence");
  return success();
}

mlir::LogicalResult DelayOp::verify() {
  auto durDeclOp = dyn_cast_or_null<mlir::arith::ConstantIntOp>(
      (*this).getDur().getDefiningOp());
  if (durDeclOp && durDeclOp.value() < 0)
    return emitOpError("duration must be >= 0.");

  if (!(*this).getTarget().isa<BlockArgument>())
    return emitOpError("Target is not a block argument; Target needs to be an "
                       "argument of pulse.sequence");

  return success();
}

//===----------------------------------------------------------------------===//
// Waveform_CreateOp
//===----------------------------------------------------------------------===//

llvm::Expected<uint64_t>
Waveform_CreateOp::getDuration(mlir::Operation *callSequenceOp = nullptr) {
  auto shape = (*this).getSamples().getType().getShape();
  if (shape[0] < 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "duration must be >= 0.");
  return shape[0];
}

/// Verifier for pulse.waveform operation.
mlir::LogicalResult Waveform_CreateOp::verify() {
  // Check that samples has two dimensions: outer is the number of
  // samples, inner is complex numbers with two elements [real, imag]
  auto attrType = getSamples().getType().cast<mlir::ShapedType>();
  auto attrShape = attrType.getShape();
  if (attrShape.size() != 2) {
    return emitOpError() << ", which declares a sample waveform, must be "
                            "composed of a two dimensional tensor.";
  }
  if (attrShape[1] != 2) {
    return emitOpError()
           << ", which declares a sample waveform, must have inner dimension "
              "two corresponding to complex elements of the form [real, imag].";
  }

  // Check duration
  auto durOrError = getDuration(nullptr /*callSequenceOp*/);
  if (auto err = durOrError.takeError())
    return emitOpError(toString(std::move(err)));

  return mlir::success();
}

//===----------------------------------------------------------------------===//
//
// end Waveform_CreateOp
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// waveform_container
//===----------------------------------------------------------------------===//

/// Verifier for pulse.waveform_container operation.
LogicalResult WaveformContainerOp::verify() {

  for (Block &block : getBody().getBlocks()) {
    for (Operation &op : block.getOperations()) {
      // Check that all the operations in the body of the waveform_container are
      // of type Waveform_CreateOp
      if (!isa<Waveform_CreateOp>(op))
        return op.emitOpError()
               << "operations other than pulse.create_waveform are not allowed "
                  "inside pulse.waveform_container.";

      // Check that Waveform_CreateOp operations has pulse.waveformName
      // attribute
      if (!op.hasAttr("pulse.waveformName"))
        return op.emitOpError()
               << "`pulse.create_waveform` operations in WaveformContainerOp "
                  "must have a `pulse.waveformName` attribute.";
    }
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
//
// end waveform_container
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CallSequenceOp
//===----------------------------------------------------------------------===//

auto CallSequenceOp::getCalleeType() -> FunctionType {
  return FunctionType::get(getContext(), getOperandTypes(), getResultTypes());
}

LogicalResult
CallSequenceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {

  auto sequenceAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!sequenceAttr)
    return emitOpError("Requires a 'callee' symbol reference attribute");

  auto sequence =
      symbolTable.lookupNearestSymbolFrom<SequenceOp>(*this, sequenceAttr);
  if (!sequence)
    return emitOpError() << "'" << sequenceAttr.getValue()
                         << "' does not reference a valid sequence";

  // Verify the types match
  auto sequenceType = sequence.getFunctionType();

  if (sequenceType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for the callee sequence");

  for (unsigned i = 0; i != sequenceType.getNumInputs(); ++i) {
    if (getOperand(i).getType() != sequenceType.getInput(i)) {
      auto diag = emitOpError("operand type mismatch at index ") << i;
      diag.attachNote() << "op input types: " << getOperandTypes();
      diag.attachNote() << "function operand types: "
                        << sequenceType.getInputs();
      return diag;
    }
  }

  if (sequenceType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for the callee sequence");

  for (unsigned i = 0; i != sequenceType.getNumResults(); ++i) {
    if (getResult(i).getType() != sequenceType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "op result types: " << getResultTypes();
      diag.attachNote() << "function result types: "
                        << sequenceType.getResults();
      return diag;
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
//
// end CallSequenceOp
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
// SequenceOp
//
// This code section was derived and modified from the LLVM project FuncOp
// Consequently it is licensed as Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

llvm::Expected<uint64_t>
SequenceOp::getDuration(mlir::Operation *callSequenceOp = nullptr) {
  // first, check if the sequence has duration attribute. If not, also check if
  // the call sequence has duration attribute; e.g., for sequences that receives
  // delay arguments, duration of the sequence can vary depending on the
  // argument, so we look at the duration of call sequence as well
  if ((*this)->hasAttr("pulse.duration"))
    return static_cast<uint64_t>(
        (*this)->getAttrOfType<IntegerAttr>("pulse.duration").getInt());
  if (callSequenceOp->hasAttr("pulse.duration"))
    return static_cast<uint64_t>(
        callSequenceOp->getAttrOfType<IntegerAttr>("pulse.duration").getInt());
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "Operation does not have a pulse.duration attribute.");
}

mlir::ParseResult SequenceOp::parse(mlir::OpAsmParser &parser,
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

void SequenceOp::print(mlir::OpAsmPrinter &printer) {
  function_interface_impl::printFunctionOp(
      printer, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
      getArgAttrsAttrName(), getResAttrsAttrName());
}

namespace {
/// Verify the argument list and entry block are in agreement.
LogicalResult verifyArgumentAndEntry_(SequenceOp op) {
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

/// Verify that no classical values are created/used in the sequence outside of
/// values that originate as argument values or the result of a measurement.
LogicalResult verifyClassical_(SequenceOp op) {
  mlir::Operation *classicalOp = nullptr;
  WalkResult const result = op->walk([&](Operation *subOp) {
    if (isa<mlir::arith::ConstantOp>(subOp) || isa<quir::ConstantOp>(subOp) ||
        isa<qcs::ParameterLoadOp>(subOp) || isa<CallSequenceOp>(subOp) ||
        isa<pulse::ReturnOp>(subOp) || isa<SequenceOp>(subOp) ||
        isa<mlir::complex::CreateOp>(subOp) ||
        subOp->hasTrait<mlir::pulse::SequenceAllowed>() ||
        subOp->hasTrait<mlir::pulse::SequenceRequired>())
      return WalkResult::advance();
    classicalOp = subOp;
    return WalkResult::interrupt();
  });

  if (result.wasInterrupted())
    return classicalOp->emitOpError()
           << "is not valid within a real-time pulse sequence.";
  return success();
}
} // anonymous namespace

LogicalResult SequenceOp::verify() {
  // If external will be linked in later and nothing to do
  if (isExternal())
    return success();

  if (failed(verifyArgumentAndEntry_(*this)))
    return mlir::failure();

  if (failed(verifyClassical_(*this)))
    return mlir::failure();

  return success();
}

SequenceOp SequenceOp::create(Location location, StringRef name,
                              FunctionType type,
                              ArrayRef<NamedAttribute> attrs) {
  OpBuilder builder(location->getContext());
  OperationState state(location, getOperationName());
  SequenceOp::build(builder, state, name, type, attrs);
  return cast<SequenceOp>(Operation::create(state));
}
SequenceOp SequenceOp::create(Location location, StringRef name,
                              FunctionType type,
                              Operation::dialect_attr_range attrs) {
  SmallVector<NamedAttribute, 8> const attrRef(attrs);
  return create(location, name, type, attrRef);
}
SequenceOp SequenceOp::create(Location location, StringRef name,
                              FunctionType type, ArrayRef<NamedAttribute> attrs,
                              ArrayRef<DictionaryAttr> argAttrs) {
  SequenceOp circ = create(location, name, type, attrs);
  circ.setAllArgAttrs(argAttrs);
  return circ;
}

void SequenceOp::build(OpBuilder &builder, OperationState &state,
                       StringRef name, FunctionType type,
                       ArrayRef<NamedAttribute> attrs,
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

/// Clone the internal blocks and attributes from this sequence to the
/// destination sequence.
void SequenceOp::cloneInto(SequenceOp dest, IRMapping &mapper) {
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

/// Create a deep copy of this sequence and all of its block.
/// Remap any operands that use values outside of the function
/// Using the provider mapper. Replace references to
/// cloned sub-values with the corresponding copied value and
/// add to the mapper
SequenceOp SequenceOp::clone(IRMapping &mapper) {
  FunctionType newType = getFunctionType();

  // If the function contains a body, then its possible arguments
  // may be deleted in the mapper. Verify this so they aren't
  // added to the input type vector.
  bool const isExternalSequence = isExternal();
  if (!isExternalSequence) {
    SmallVector<Type, 4> inputTypes;
    inputTypes.reserve(newType.getNumInputs());
    for (unsigned i = 0; i != getNumArguments(); ++i)
      if (!mapper.contains(getArgument(i)))
        inputTypes.push_back(newType.getInput(i));
    newType = FunctionType::get(getContext(), inputTypes, newType.getResults());
  }

  // Create the new sequence
  SequenceOp newSeq = cast<SequenceOp>(getOperation()->cloneWithoutRegions());
  newSeq.setType(newType);

  // Clone the current function into the new one and return.
  cloneInto(newSeq, mapper);
  return newSeq;
}

SequenceOp SequenceOp::clone() {
  IRMapping mapper;
  return clone(mapper);
}

//===----------------------------------------------------------------------===//
//
// end SequenceOp

//===----------------------------------------------------------------------===//
// ReturnOp
//
// This code section was derived and modified from the LLVM project's standard
// dialect ReturnOp. Consequently it is licensed as described below.
//
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  auto sequence = (*this)->getParentOfType<SequenceOp>();
  FunctionType const sequenceType = sequence.getFunctionType();

  auto numResults = sequenceType.getNumResults();
  // Verify number of operands match type signature
  if (numResults != getOperands().size()) {
    return emitError()
        .append("expected ", numResults, " result operands")
        .attachNote(sequence.getLoc())
        .append("return type declared here");
  }

  int i = 0;
  for (const auto [type, operand] :
       llvm::zip(sequenceType.getResults(), getOperands())) {
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
// PlayOp
//===----------------------------------------------------------------------===//

llvm::Expected<uint64_t>
PlayOp::getDuration(mlir::Operation *callSequenceOp = nullptr) {

  // check if callSequenceOp arg is specified and if not, return the value of
  // pulse.duration attribute
  auto callOp = dyn_cast_or_null<CallSequenceOp>(callSequenceOp);
  if (!callOp) {
    if ((*this)->hasAttr("pulse.duration"))
      return static_cast<uint64_t>(
          (*this)->getAttrOfType<IntegerAttr>("pulse.duration").getInt());
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Operation does not have a pulse.duration attribute, and no "
        "callSequenceOp argument is specified.");
  }

  // check if wfr is of type Waveform_CreateOp; this is the case if wfrOp is
  // defined in the same block as playOp e.g., if both are defined inside a
  // sequenceOp
  if (auto castOp = dyn_cast_or_null<Waveform_CreateOp>(
          (*this).getWfr().getDefiningOp())) {
    llvm::Expected<uint64_t> durOrError =
        castOp.getDuration(nullptr /*callSequenceOp*/);
    if (auto err = durOrError.takeError())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     toString(std::move(err)));
    return durOrError.get();
  }

  auto argIndex = (*this).getWfr().cast<BlockArgument>().getArgNumber();
  auto *argOp = callOp->getOperand(argIndex).getDefiningOp();
  auto wfrOp = dyn_cast_or_null<Waveform_CreateOp>(argOp);
  if (wfrOp) {
    llvm::Expected<uint64_t> durOrError =
        wfrOp.getDuration(nullptr /*callSequenceOp*/);
    if (auto err = durOrError.takeError())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     toString(std::move(err)));
    return durOrError.get();
  }
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "Could not get the wfrOp!");
}

llvm::Expected<std::string> PlayOp::getWaveformHash(CallSequenceOp callOp) {
  // attempt to get waveform and target directly from PlayOp if that
  // fails get to the defining op via the call sequence

  Operation *wfrOp;
  Operation *targetOp;
  wfrOp = dyn_cast_or_null<Waveform_CreateOp>(getWfr().getDefiningOp());
  targetOp = dyn_cast_or_null<MixFrameOp>(getTarget().getDefiningOp());

  if (!wfrOp && !targetOp) {
    auto wfrArgIndex = getWfr().dyn_cast<BlockArgument>().getArgNumber();
    wfrOp = callOp.getOperand(wfrArgIndex)
                .getDefiningOp<mlir::pulse::Waveform_CreateOp>();
    auto mixFrameArgIndex =
        getTarget().dyn_cast<BlockArgument>().getArgNumber();
    targetOp = callOp.getOperand(mixFrameArgIndex)
                   .getDefiningOp<mlir::pulse::MixFrameOp>();
  }

  if (wfrOp && targetOp) {
    auto targetHash = mlir::hash_value(targetOp->getLoc());
    auto wfrHash = mlir::hash_value(wfrOp->getLoc());
    return std::to_string(targetHash) + "_" + std::to_string(wfrHash);
  }

  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "Failed to hash waveform name from play operation");
}

mlir::LogicalResult PlayOp::verify() {
  if (!(*this).getTarget().isa<BlockArgument>())
    return emitOpError("Target is not a block argument; Target needs to be an "
                       "argument of pulse.sequence");
  return success();
}

//===----------------------------------------------------------------------===//
//
// end PlayOp
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// end Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

} // namespace mlir::pulse

#define GET_OP_CLASSES
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/Pulse/IR/Pulse.cpp.inc"
