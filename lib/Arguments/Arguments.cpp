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
#include "API/errors.h"
#include "Payload/Payload.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"

#include <llvm/Support/raw_ostream.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <utility>

namespace qssc::arguments {

using namespace payload;

llvm::Error updateParameters(qssc::payload::PatchablePayload *payload,
                             Signature &sig, ArgumentSource const &arguments,
                             bool treatWarningsAsErrors,
                             BindArgumentsImplementationFactory &factory,
                             const OptDiagnosticCallback &onDiagnostic) {

  for (const auto &[binaryName, patchPoints] : sig.patchPointsByBinary) {

    if (patchPoints.size() == 0) // no patch points
      continue;

    auto binaryDataOrErr = payload->readMember(binaryName);

    if (!binaryDataOrErr) {
      auto error = binaryDataOrErr.takeError();
      return emitDiagnostic(onDiagnostic, qssc::Severity::Error,
                            qssc::ErrorCategory::QSSLinkSignatureError,
                            "Error reading " + binaryName + " " +
                                toString(std::move(error)));
    }

    auto &binaryData = binaryDataOrErr.get();

    auto binary = std::shared_ptr<BindArgumentsImplementation>(
        factory.create(binaryData, onDiagnostic));
    binary->setTreatWarningsAsErrors(treatWarningsAsErrors);

    for (auto const &patchPoint : patchPoints)
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

  // placeholder string for data on disk if required
  std::string inputFromDisk;

  if (!enableInMemoryInput) {
    // compile payload on disk
    // copy to link payload if not returning in memory
    // load from disk into string if returning in memory
    if (!enableInMemoryOutput) {
      std::error_code copyError =
          llvm::sys::fs::copy_file(moduleInput, payloadOutputPath);
      if (copyError)
        return llvm::make_error<llvm::StringError>(
            "Failed to copy circuit module to payload", copyError);
    } else {
      // read from disk to process in memory
      std::ostringstream buf;
      std::ifstream input(moduleInput.str().c_str());
      buf << input.rdbuf();
      inputFromDisk = buf.str();
      moduleInput = inputFromDisk;
      enableInMemoryInput = true;
    }
  }

  if (!enableInMemoryOutput && enableInMemoryInput) {
    // if payload in memory but returning on disk
    // copy to disk and process from there
    std::ofstream payload;
    payload.open(payloadOutputPath.str(), std::ios::binary);
    payload.write(moduleInput.str().c_str(), moduleInput.str().length());
    payload.close();
    enableInMemoryInput = false;
  }

  llvm::StringRef payloadData =
      (enableInMemoryInput) ? moduleInput : payloadOutputPath;

  auto binary = std::unique_ptr<BindArgumentsImplementation>(
      factory.create(onDiagnostic));
  binary->setTreatWarningsAsErrors(treatWarningsAsErrors);

  auto payload = std::unique_ptr<PatchablePayload>(
      binary->getPayload(payloadData, enableInMemoryInput));

  auto sigOrError = binary->parseSignature(payload.get());
  if (auto err = sigOrError.takeError())
    return err;

  if (auto err = updateParameters(payload.get(), sigOrError.get(), arguments,
                                  treatWarningsAsErrors, factory, onDiagnostic))
    return err;

  // setup linked payload I/O
  // if enableInMemoryOutput is true:
  //    write to string
  // if enableInMemoryInput is true:
  //    payload is not on disk yet, do not assume payload->writeBack()
  //    will write the full payload to disk so: write to string,
  //    dump string to disk and clear string
  // if enableInMemoryInput is false:
  //    payload was on disk originally use writeBack
  if (auto err = payload->writeBack())
    return err;
  if (enableInMemoryOutput || enableInMemoryInput) {
    if (auto err = payload->writeString(inMemoryOutput))
      return err;
    if (!enableInMemoryOutput) {
      auto pathStr = payloadOutputPath.str();
      std::ofstream out(pathStr);
      out << inMemoryOutput;
      out.close();
      // clear output string
      *inMemoryOutput = "";
    }
  }

  return llvm::Error::success();
}

} // namespace qssc::arguments
