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
#include "Dialect/Pulse/IR/PulseDialect.h"

#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/Twine.h"

namespace mlir::pulse {

//===----------------------------------------------------------------------===//
// Waveform_CreateOp
//===----------------------------------------------------------------------===//

llvm::Expected<uint64_t> Waveform_CreateOp::getPulseOpDuration(
    mlir::Operation *callSequenceOp = nullptr) {
  auto shape = (*this).samples().getType().getShape();
  if (shape[0] < 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "duration must be >= 0.");
  return shape[0];
}

/// Verifier for pulse.waveform operation.
static auto verify(Waveform_CreateOp &op) -> mlir::LogicalResult {
  // Check that samples is a tensor with two dimensions: outer is the number of
  // samples, inner is complex numbers with two elements [real, imag]
  auto attrType = op.samples().getType().cast<mlir::TensorType>();
  auto attrShape = attrType.getShape();
  if (attrShape.size() != 2) {
    return op.emitOpError() << ", which declares a sample waveform, must be "
                               "composed of a two dimensional tensor.";
  }
  if (attrShape[1] != 2) {
    return op.emitOpError()
           << ", which declares a sample waveform, must have inner dimension "
              "two corresponding to complex elements of the form [real, imag].";
  }

  // Check duration
  auto durOrError = op.getPulseOpDuration(nullptr /*callSequenceOp*/);
  if (auto err = durOrError.takeError())
    return op.emitOpError(toString(std::move(err)));

  return mlir::success();
}

//===----------------------------------------------------------------------===//
//
// end Waveform_CreateOp
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
  auto sequenceType = sequence.getType();

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

static ParseResult parseSequenceOp(OpAsmParser &parser,
                                   OperationState &result) {
  auto buildSequenceType =
      [](Builder &builder, ArrayRef<Type> argTypes, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };
  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false, buildSequenceType);
}

static void print(SequenceOp op, OpAsmPrinter &p) {
  FunctionType fnType = op.getType();
  function_interface_impl::printFunctionOp(
      p, op, fnType.getInputs(), /*isVariadic=*/false, fnType.getResults());
}

/// Verify the argument list and entry block are in agreement.
static LogicalResult verifyArgumentAndEntry_(SequenceOp op) {
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

/// Verify that no classical values are created/used in the sequence outside of
/// values that originate as argument values or the result of a measurement.
static LogicalResult verifyClassical_(SequenceOp op) {
  mlir::Operation *classicalOp = nullptr;
  WalkResult result = op->walk([&](Operation *subOp) {
    if (isa<mlir::ConstantOp>(subOp) || isa<mlir::arith::ConstantOp>(subOp) ||
        isa<quir::ConstantOp>(subOp) || isa<CallSequenceOp>(subOp) ||
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

static LogicalResult verify(SequenceOp op) {
  // If external will be linked in later and nothing to do
  if (op.isExternal())
    return success();

  if (failed(verifyArgumentAndEntry_(op)))
    return mlir::failure();

  if (failed(verifyClassical_(op)))
    return mlir::failure();

  return success();
}

/// Clone the internal blocks and attributes from this sequence to the
/// destination sequence.
void SequenceOp::cloneInto(SequenceOp dest, BlockAndValueMapping &mapper) {
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
SequenceOp SequenceOp::clone(BlockAndValueMapping &mapper) {
  FunctionType newType = getType();

  // If the function contains a body, then its possible arguments
  // may be deleted in the mapper. Verify this so they aren't
  // added to the input type vector.
  bool isExternalSequence = isExternal();
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
  BlockAndValueMapping mapper;
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

static LogicalResult verify(ReturnOp op) {
  auto sequence = op->getParentOfType<SequenceOp>();
  FunctionType sequenceType = sequence.getType();

  auto numResults = sequenceType.getNumResults();
  // Verify number of operands match type signature
  if (numResults != op.operands().size()) {
    return op.emitError()
        .append("expected ", numResults, " result operands")
        .attachNote(sequence.getLoc())
        .append("return type declared here");
  }

  int i = 0;
  for (const auto [type, operand] :
       llvm::zip(sequenceType.getResults(), op.operands())) {
    auto opType = operand.getType();
    if (type != opType) {
      return op.emitOpError()
             << "unexpected type `" << opType << "' for operand #" << i;
    }
    i++;
  }
  return success();
}

llvm::Optional<int64_t> ReturnOp::getTimepoint() {
  if ((*this)->hasAttr("pulse.timepoint"))
    return (*this)->getAttrOfType<IntegerAttr>("pulse.timepoint").getInt();
  return llvm::None;
}

void ReturnOp::setTimepoint(int64_t timepoint) {
  mlir::OpBuilder builder(*this);
  (*this)->setAttr("pulse.timepoint", builder.getI64IntegerAttr(timepoint));
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
PlayOp::getPulseOpDuration(mlir::Operation *callSequenceOp = nullptr) {

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
  if (auto castOp =
          dyn_cast_or_null<Waveform_CreateOp>((*this).wfr().getDefiningOp())) {
    llvm::Expected<uint64_t> durOrError =
        castOp.getPulseOpDuration(nullptr /*callSequenceOp*/);
    if (auto err = durOrError.takeError())
      return err;
    return durOrError.get();
  }

  auto argIndex = (*this).wfr().cast<BlockArgument>().getArgNumber();
  auto *argOp = callOp->getOperand(argIndex).getDefiningOp();
  auto wfrOp = dyn_cast_or_null<Waveform_CreateOp>(argOp);
  if (wfrOp) {
    llvm::Expected<uint64_t> durOrError =
        wfrOp.getPulseOpDuration(nullptr /*callSequenceOp*/);
    if (auto err = durOrError.takeError())
      return err;
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
  wfrOp = dyn_cast_or_null<Waveform_CreateOp>(wfr().getDefiningOp());
  targetOp = dyn_cast_or_null<MixFrameOp>(target().getDefiningOp());

  if (!wfrOp && !targetOp) {
    auto wfrArgIndex = wfr().dyn_cast<BlockArgument>().getArgNumber();
    wfrOp = callOp.getOperand(wfrArgIndex)
                .getDefiningOp<mlir::pulse::Waveform_CreateOp>();
    auto mixFrameArgIndex = target().dyn_cast<BlockArgument>().getArgNumber();
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

llvm::Optional<int64_t> PlayOp::getTimepoint() {
  if ((*this)->hasAttr("pulse.timepoint"))
    return (*this)->getAttrOfType<IntegerAttr>("pulse.timepoint").getInt();
  return llvm::None;
}

void PlayOp::setTimepoint(int64_t timepoint) {
  mlir::OpBuilder builder(*this);
  (*this)->setAttr("pulse.timepoint", builder.getI64IntegerAttr(timepoint));
}

llvm::Optional<uint64_t> PlayOp::getSetupLatency() {
  if ((*this)->hasAttr("pulse.setupLatency"))
    return static_cast<uint64_t>(
        (*this)->getAttrOfType<IntegerAttr>("pulse.setupLatency").getInt());
  return llvm::None;
}

void PlayOp::setSetupLatency(uint64_t setupLatency) {
  mlir::OpBuilder builder(*this);
  (*this)->setAttr("pulse.setupLatency",
                   builder.getI64IntegerAttr(setupLatency));
}

//===----------------------------------------------------------------------===//
//
// end PlayOp
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CaptureOp
//===----------------------------------------------------------------------===//

llvm::Optional<int64_t> CaptureOp::getTimepoint() {
  if ((*this)->hasAttr("pulse.timepoint"))
    return (*this)->getAttrOfType<IntegerAttr>("pulse.timepoint").getInt();
  return llvm::None;
}

void CaptureOp::setTimepoint(int64_t timepoint) {
  mlir::OpBuilder builder(*this);
  (*this)->setAttr("pulse.timepoint", builder.getI64IntegerAttr(timepoint));
}

llvm::Optional<uint64_t> CaptureOp::getSetupLatency() {
  if ((*this)->hasAttr("pulse.setupLatency"))
    return static_cast<uint64_t>(
        (*this)->getAttrOfType<IntegerAttr>("pulse.setupLatency").getInt());
  return llvm::None;
}

void CaptureOp::setSetupLatency(uint64_t setupLatency) {
  mlir::OpBuilder builder(*this);
  (*this)->setAttr("pulse.setupLatency",
                   builder.getI64IntegerAttr(setupLatency));
}

//===----------------------------------------------------------------------===//
//
// end CaptureOp
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// DelayOp
//===----------------------------------------------------------------------===//

llvm::Expected<uint64_t>
DelayOp::getPulseOpDuration(mlir::Operation *callSequenceOp = nullptr) {
  // get op that defines duration and return its value
  auto durDeclOp = dyn_cast_or_null<mlir::arith::ConstantIntOp>(
      (*this).dur().getDefiningOp());
  if (durDeclOp) {
    if (durDeclOp.value() < 0)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "duration must be >= 0.");
    return durDeclOp.value();
  }
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "Could not get the value of the op that defines duration!");
}

//===----------------------------------------------------------------------===//
//
// end DelayOp
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// GaussianOp
//===----------------------------------------------------------------------===//

llvm::Expected<uint64_t>
GaussianOp::getPulseOpDuration(mlir::Operation *callSequenceOp = nullptr) {
  // get op that defines duration and return its value
  auto durDeclOp = dyn_cast_or_null<mlir::arith::ConstantIntOp>(
      (*this).dur().getDefiningOp());
  if (durDeclOp) {
    if (durDeclOp.value() < 0)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "duration must be >= 0.");
    return durDeclOp.value();
  }
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "Could not get the value of the op that defines duration!");
}

//===----------------------------------------------------------------------===//
//
// end GaussianOp
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// GaussianSquareOp
//===----------------------------------------------------------------------===//

llvm::Expected<uint64_t> GaussianSquareOp::getPulseOpDuration(
    mlir::Operation *callSequenceOp = nullptr) {
  // get op that defines duration and return its value
  auto durDeclOp = dyn_cast_or_null<mlir::arith::ConstantIntOp>(
      (*this).dur().getDefiningOp());
  if (durDeclOp) {
    if (durDeclOp.value() < 0)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "duration must be >= 0.");
    return durDeclOp.value();
  }
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "Could not get the value of the op that defines duration!");
}

//===----------------------------------------------------------------------===//
//
// end GaussianSquareOp
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// DragOp
//===----------------------------------------------------------------------===//

llvm::Expected<uint64_t>
DragOp::getPulseOpDuration(mlir::Operation *callSequenceOp = nullptr) {
  // get op that defines duration and return its value
  auto durDeclOp = dyn_cast_or_null<mlir::arith::ConstantIntOp>(
      (*this).dur().getDefiningOp());
  if (durDeclOp) {
    if (durDeclOp.value() < 0)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "duration must be >= 0.");
    return durDeclOp.value();
  }
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "Could not get the value of the op that defines duration!");
}

//===----------------------------------------------------------------------===//
//
// end DragOp
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstOp
//===----------------------------------------------------------------------===//

llvm::Expected<uint64_t>
ConstOp::getPulseOpDuration(mlir::Operation *callSequenceOp = nullptr) {
  // get op that defines duration and return its value
  auto durDeclOp = dyn_cast_or_null<mlir::arith::ConstantIntOp>(
      (*this).dur().getDefiningOp());
  if (durDeclOp) {
    if (durDeclOp.value() < 0)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "duration must be >= 0.");
    return durDeclOp.value();
  }
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "Could not get the value of the op that defines duration!");
}

//===----------------------------------------------------------------------===//
//
// end ConstOp
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// end Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

} // namespace mlir::pulse

#define GET_OP_CLASSES
#include "Dialect/Pulse/IR/Pulse.cpp.inc"
