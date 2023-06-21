//===- Signature.cpp --------------------------------------------*- C++ -*-===//
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
///
///  This file implements the Parameter Signature of a circuit module for
///  updating arguments after compilation.
///
//===----------------------------------------------------------------------===//

#include "Arguments/Signature.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>

namespace qssc::arguments {

void Signature::addParameterPatchPoint(llvm::StringRef expression,
                                       llvm::StringRef patchType,
                                       llvm::StringRef binaryComponent,
                                       uint64_t offset) {

  auto &patchPoints = patchPointsByBinary[binaryComponent];

  patchPoints.emplace_back(expression, patchType, offset);
}

void Signature::addParameterPatchPoint(llvm::StringRef binaryComponent,
                                       PatchPoint p) {

  auto &patchPoints = patchPointsByBinary[binaryComponent];
  patchPoints.push_back(p);
}

void Signature::dump() {
  llvm::errs() << "Circuit Signature:\n";

  for (auto const &entry : patchPointsByBinary) {

    llvm::errs() << "binary " << entry.getKey() << ":\n";

    for (auto const &patchPoint : entry.getValue()) {
      llvm::errs() << "  param expression " << patchPoint.expression()
                   << " to be patched as " << patchPoint.patchType()
                   << " at offset " << patchPoint.offset() << "\n";
    }
  }
}

std::string Signature::serialize() {
  std::stringstream s;
  s << "circuit_signature\n";
  s << "version 1\n";
  s << "num_binaries: " << patchPointsByBinary.size() << "\n";

  for (auto const &entry : patchPointsByBinary) {
    auto patchPoints = entry.getValue();
    s << "binary: " << entry.getKey().str() << "\n";
    s << "num_patchpoints: " << patchPoints.size() << "\n";
    for (auto const &patchPoint : patchPoints) {
      s << patchPoint.patchType().str() << " " << patchPoint.offset() << " "
        << patchPoint.expression().str() << "\n";
    }
  }
  return s.str();
}

llvm::Expected<Signature> Signature::deserialize(llvm::StringRef buffer) {

  Signature sig;

  llvm::StringRef line;

  std::tie(line, buffer) = buffer.split("\n");
  if (line != "circuit_signature") {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Error: Invalid Signature header\n");
  }

  std::tie(line, buffer) = buffer.split("\n");
  if (line != "version 1") {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Error: Invalid Signature version\n");
  }

  std::tie(line, buffer) = buffer.split("\n");

  llvm::StringRef label;
  llvm::StringRef value;

  std::tie(label, value) = line.split(' ');
  if (label != "num_binaries:") {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Error: expected num_binaries:\n");
  }
  uint numBinaries;
  if (value.getAsInteger(10, numBinaries)) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Error: failed to parse number of binaries to integer: " + value.str() +
            "\n");
  }

  for (uint nBinary = 0; nBinary < numBinaries; nBinary++) {
    std::tie(line, buffer) = buffer.split("\n");
    std::tie(label, value) = line.split(' ');
    if (label != "binary:") {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Error expected binary:\n");
    }
    llvm::StringRef binaryName = value;

    std::tie(line, buffer) = buffer.split("\n");
    std::tie(label, value) = line.split(' ');
    if (label != "num_patchpoints:") {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Error: expected num_patchpoints:\n");
    }
    uint numEntries;
    if (value.getAsInteger(10, numEntries)) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "Error: failed to parse number of entries to integer: " +
              value.str() + "\n");
    }
    for (uint nEntry = 0; nEntry < numEntries; nEntry++) {
      std::tie(line, buffer) = buffer.split("\n");
      llvm::SmallVector<llvm::StringRef, 3> components;
      line.split(components, ' ', 2, false);
      if (components.size() != 3) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "invalid argument entry line: " + line +
                                           "\n");
      }
      uint64_t addr;
      if (components[1].getAsInteger(10, addr)) {
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "failed to parse address to integer: " +
                                           components[1].str() + "\n");
      }

      auto paramPatchType = components[0];
      auto expression = components[2];

      sig.addParameterPatchPoint(expression, paramPatchType, binaryName, addr);
    }
  }

  if (buffer.size() > 0)
    llvm::errs() << "ignoring extra data at end of signature file\n";
  return sig;
}

} // namespace qssc::arguments
