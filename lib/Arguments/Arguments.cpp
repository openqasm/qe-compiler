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
#include "Payload/Payload.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"

#include <memory>
#include <utility>

namespace qssc::arguments {

using namespace payload;

llvm::Error updateParameters(qssc::payload::PatchablePayload *payload,
                             Signature &sig, ArgumentSource const &arguments,
                             BindArgumentsImplementationFactory *factory) {

  for (auto &entry : sig.patchPointsByBinary) {
    auto binaryName = entry.getKey();
    auto patchPoints = entry.getValue();

    if (patchPoints.size() == 0) // no patch points
      continue;

    auto binaryDataOrErr = payload->readMember(binaryName);

    if (!binaryDataOrErr)
      return binaryDataOrErr.takeError();

    auto &binaryData = binaryDataOrErr.get();

    auto binary = std::shared_ptr<BindArgumentsImplementation>(
        factory->create(binaryData));

    for (auto const &patchPoint : entry.getValue())
      if (auto err = binary->patch(patchPoint, arguments))
        return err;
  }

  return llvm::Error::success();
}

llvm::Error bindArguments(llvm::StringRef moduleInputPath,
                          llvm::StringRef payloadOutputPath,
                          ArgumentSource const &arguments,
                          BindArgumentsImplementationFactory *factory) {

  std::error_code copyError =
      llvm::sys::fs::copy_file(moduleInputPath, payloadOutputPath);

  if (copyError)
    return llvm::make_error<llvm::StringError>(
        "Failed to copy circuit module to payload", copyError);

  auto binary = std::unique_ptr<BindArgumentsImplementation>(factory->create());
  auto payload =
      std::unique_ptr<PatchablePayload>(binary->getPayload(payloadOutputPath));
  auto sigOrError = binary->parseSignature(payload.get());
  if (auto err = sigOrError.takeError())
    return err;

  if (auto err =
          updateParameters(payload.get(), sigOrError.get(), arguments, factory))
    return err;

  if (auto err = payload->writeBack())
    return err;

  return llvm::Error::success();
}

} // namespace qssc::arguments
