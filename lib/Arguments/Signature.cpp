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

#include "API/errors.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <tuple>

namespace qssc::arguments {

void Signature::addParameterPatchPoint(llvm::StringRef expression,
                                       llvm::StringRef patchType,
                                       llvm::StringRef binaryComponent,
                                       uint64_t offset) {

  auto &patchPoints = patchPointsByBinary[binaryComponent.str()];

  patchPoints.emplace_back(expression, patchType, offset);
}

void Signature::addParameterPatchPoint(llvm::StringRef binaryComponent,
                                       const PatchPoint &p) {

  auto &patchPoints = patchPointsByBinary[binaryComponent.str()];
  patchPoints.push_back(p);
}

void Signature::dump() {
  llvm::errs() << "Circuit Signature:\n";

  for (auto const &[binaryName, patchPoints] : patchPointsByBinary) {

    llvm::errs() << "binary " << binaryName << ":\n";

    for (auto const &patchPoint : patchPoints) {
      llvm::errs() << "  param expression " << patchPoint.expression()
                   << " to be patched as " << patchPoint.patchType()
                   << " at offset " << patchPoint.offset() << "\n";
    }
  }
}

std::string Signature::serialize() const {
  std::stringstream s;
  s << "circuit_signature\n";
  s << "version 1\n";
  s << "num_binaries: " << patchPointsByBinary.size() << "\n";

  for (auto const &[binaryName, patchPoints] : patchPointsByBinary) {
    s << "binary: " << binaryName << "\n";
    s << "num_patchpoints: " << patchPoints.size() << "\n";
    for (auto const &patchPoint : patchPoints) {
      s << patchPoint.patchType().str() << " " << patchPoint.offset() << " "
        << patchPoint.expression().str() << "\n";
    }
  }
  return s.str();
}

llvm::Expected<Signature>
Signature::deserialize(llvm::StringRef buffer,
                       const qssc::OptDiagnosticCallback &onDiagnostic,
                       bool treatWarningsAsErrors) {

  Signature sig;

  llvm::StringRef line;

  std::tie(line, buffer) = buffer.split("\n");
  if (line != "circuit_signature") {
    return emitDiagnostic(onDiagnostic, qssc::Severity::Error,
                          qssc::ErrorCategory::QSSLinkSignatureError,
                          "Invalid Signature header");
  }

  std::tie(line, buffer) = buffer.split("\n");
  if (line != "version 1") {
    return emitDiagnostic(onDiagnostic, qssc::Severity::Error,
                          qssc::ErrorCategory::QSSLinkSignatureError,
                          "Invalid Signature version: " + line.str());
  }

  std::tie(line, buffer) = buffer.split("\n");

  llvm::StringRef label;
  llvm::StringRef value;

  std::tie(label, value) = line.split(' ');
  if (label != "num_binaries:") {
    return emitDiagnostic(onDiagnostic, qssc::Severity::Error,
                          qssc::ErrorCategory::QSSLinkSignatureError,
                          "Expected num_binaries:");
  }
  uint numBinaries;
  if (value.getAsInteger(10, numBinaries)) {
    return emitDiagnostic(onDiagnostic, qssc::Severity::Error,
                          qssc::ErrorCategory::QSSLinkSignatureError,
                          "Failed to parse number of binaries to integer: " +
                              value.str());
  }

  for (uint nBinary = 0; nBinary < numBinaries; nBinary++) {
    std::tie(line, buffer) = buffer.split("\n");
    std::tie(label, value) = line.split(' ');
    if (label != "binary:") {
      return emitDiagnostic(onDiagnostic, qssc::Severity::Error,
                            qssc::ErrorCategory::QSSLinkSignatureError,
                            "Expected binary:");
    }
    llvm::StringRef const binaryName = value;

    std::tie(line, buffer) = buffer.split("\n");
    std::tie(label, value) = line.split(' ');
    if (label != "num_patchpoints:") {
      return emitDiagnostic(onDiagnostic, qssc::Severity::Error,
                            qssc::ErrorCategory::QSSLinkSignatureError,
                            "Expected num_patchpoints:");
    }
    uint numEntries;
    if (value.getAsInteger(10, numEntries)) {
      return emitDiagnostic(onDiagnostic, qssc::Severity::Error,
                            qssc::ErrorCategory::QSSLinkSignatureError,
                            "Failed to parse number of entries to integer: " +
                                value.str());
    }
    for (uint nEntry = 0; nEntry < numEntries; nEntry++) {
      std::tie(line, buffer) = buffer.split("\n");
      llvm::SmallVector<llvm::StringRef, 3> components;
      line.split(components, ' ', 2, false);
      if (components.size() != 3) {
        return emitDiagnostic(onDiagnostic, qssc::Severity::Error,
                              qssc::ErrorCategory::QSSLinkSignatureError,
                              "Invalid argument entry line: " + line.str());
      }
      uint64_t addr;
      if (components[1].getAsInteger(10, addr)) {
        return emitDiagnostic(onDiagnostic, qssc::Severity::Error,
                              qssc::ErrorCategory::QSSLinkAddressError,
                              "Failed to interpret argument address " +
                                  components[1].str());
      }

      auto paramPatchType = components[0];
      auto expression = components[2];

      sig.addParameterPatchPoint(expression, paramPatchType, binaryName, addr);
    }
  }

  if (buffer.size() > 0) {
    if (!treatWarningsAsErrors) {
      // cast to void to discard llvm::Error
      static_cast<void>(
          emitDiagnostic(onDiagnostic, qssc::Severity::Warning,
                         qssc::ErrorCategory::QSSLinkSignatureWarning,
                         "Ignoring extra data at end of signature file"));
    } else {
      return emitDiagnostic(onDiagnostic, qssc::Severity::Error,
                            qssc::ErrorCategory::QSSLinkSignatureWarning,
                            "Ignoring extra data at end of signature file");
    }
  }
  return sig;
}

} // namespace qssc::arguments
