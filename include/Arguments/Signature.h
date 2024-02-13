//===- Signature.h ----------------------------------------------*- C++ -*-===//
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
///  This file declares the Signature of a circuit module, that is, the
///  arguments accepted by the type and location information about where they
///  need to be patched in the module.
///
//===----------------------------------------------------------------------===//

#ifndef ARGUMENTS_SIGNATURE_H
#define ARGUMENTS_SIGNATURE_H

#include "API/errors.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace qssc::arguments {

class PatchPoint {
  std::string expression_;
  std::string patchType_;
  // TODO we will have more types of patch points, need more flexible structure
  // for parameters
  uint64_t offset_;

public:
  PatchPoint(llvm::StringRef expression, llvm::StringRef patchType,
             uint64_t offset)
      : expression_(expression), patchType_(patchType), offset_(offset) {}

  friend struct Signature;

public:
  llvm::StringRef expression() const { return expression_; }
  llvm::StringRef patchType() const { return patchType_; }
  uint64_t offset() const { return offset_; }
};

using PatchPointVector = std::vector<PatchPoint>;

struct Signature {
  // TODO consider deduplicating strings by using UniqueStringSaver
  // Use std::map instead of StringMap to preserve order
  std::map<std::string, std::vector<PatchPoint>> patchPointsByBinary;

public:
  void addParameterPatchPoint(llvm::StringRef expression,
                              llvm::StringRef patchType,
                              llvm::StringRef binaryComponent, uint64_t offset);
  void addParameterPatchPoint(llvm::StringRef binaryComponent,
                              const PatchPoint &p);
  void dump();

  std::string serialize() const;

  static llvm::Expected<Signature>
  deserialize(llvm::StringRef, const qssc::OptDiagnosticCallback &onDiagnostic,
              bool treatWarningsAsError = false);

  bool isEmpty() const { return patchPointsByBinary.size() == 0; }
};

} // namespace qssc::arguments

#endif // PARAMETER_SIGNATURE_H
