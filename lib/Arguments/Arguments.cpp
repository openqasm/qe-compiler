//===- Arguments.cpp -------------------------------------------*- C++ -*-===//
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
///  This file implements support for updating argument values after compilation
///
//===----------------------------------------------------------------------===//

#include "Arguments/Arguments.h"
#include "API/error.h"
#include "Payload/Payload.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"

#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <utility>

namespace qssc::arguments {

using namespace payload;

llvm::Error updateParameters(qssc::payload::PatchablePayload *payload,
                             Signature &sig, ArgumentSource const &arguments,
                             bool treatWarningsAsErrors,
                             BindArgumentsImplementationFactory &factory,
                             const OptDiagnosticCallback &onDiagnostic) {

  for (auto &entry : sig.patchPointsByBinary) {
    auto binaryName = entry.getKey();
    auto patchPoints = entry.getValue();

    if (patchPoints.size() == 0) // no patch points
      continue;

    auto binaryDataOrErr = payload->readMember(binaryName);

    if (!binaryDataOrErr) {
      auto error = binaryDataOrErr.takeError();
      return emitDiagnostic(onDiagnostic, qssc::Severity::Error,
                            qssc::ErrorCategory::QSSLinkSignatureError,
                            "Error reading " + binaryName.str() + " " +
                                toString(std::move(error)));
    }

    auto &binaryData = binaryDataOrErr.get();

    auto binary = std::shared_ptr<BindArgumentsImplementation>(
        factory.create(binaryData, onDiagnostic));
    binary->setTreatWarningsAsErrors(treatWarningsAsErrors);

    for (auto const &patchPoint : entry.getValue())
      if (auto err = binary->patch(patchPoint, arguments))
        return err;
  }

  return llvm::Error::success();
}

llvm::Error bindArguments(llvm::StringRef moduleInput,
                          llvm::StringRef payloadOutputPath,
                          ArgumentSource const &arguments,
                          bool treatWarningsAsErrors, bool enableInMemoryInput,
                          std::string *inMemoryOutput,
                          BindArgumentsImplementationFactory &factory,
                          const OptDiagnosticCallback &onDiagnostic) {

  bool enableInMemoryOutput = payloadOutputPath == "";

  llvm::StringRef payloadPath =
      (enableInMemoryOutput) ? moduleInput : payloadOutputPath;

  if (!enableInMemoryOutput) {
    std::error_code copyError =
        llvm::sys::fs::copy_file(moduleInput, payloadOutputPath);

    if (copyError)
      return llvm::make_error<llvm::StringError>(
          "Failed to copy circuit module to payload", copyError);
  }

  auto binary = std::unique_ptr<BindArgumentsImplementation>(
      factory.create(onDiagnostic));
  binary->setTreatWarningsAsErrors(treatWarningsAsErrors);

  auto payload =
      std::unique_ptr<PatchablePayload>(binary->getPayload(payloadPath));

  auto sigOrError = binary->parseSignature(payload.get());
  if (auto err = sigOrError.takeError())
    return err;

  if (auto err = updateParameters(payload.get(), sigOrError.get(), arguments,
                                  treatWarningsAsErrors, factory, onDiagnostic))
    return err;

  if (enableInMemoryOutput) {
    if (auto err = payload->writeString(inMemoryOutput))
      return err;
  } else if (auto err = payload->writeBack())
    return err;

  return llvm::Error::success();
}

} // namespace qssc::arguments
