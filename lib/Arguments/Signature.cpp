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

namespace qssc::arguments {

void Signature::addParameterPatchPoint(llvm::StringRef expression,
                                       llvm::StringRef patchType,
                                       llvm::StringRef binaryComponent,
                                       uint64_t offset) {

  auto &patchPoints = patchPointsByBinary[binaryComponent];

  patchPoints.emplace_back(expression, patchType, offset);
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

std::string Signature::serialize() { return ""; }

} // namespace qssc::arguments
