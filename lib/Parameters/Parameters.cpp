//===- Parameters.cpp -------------------------------------------*- C++ -*-===//
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
///  This file implements parameter binding.
///
//===----------------------------------------------------------------------===//

#include "Parameters/Parameters.h"
#include "Payload/PatchableZipPayload.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"

#include <utility>

namespace qssc::parameters {

using namespace payload;

// TODO move to payload utilities
static llvm::Expected<std::string> readFileFromZip(zip_t *zip, zip_stat_t &zs) {
  auto *zipFile = zip_fopen_index(zip, zs.index, 0);

  if (!zipFile) {
    auto *err = zip_get_error(zip);
    return extractLibZipError("Opening file within zip", *err);
  }

  std::string fileBuf(zs.size, '\0');

  if (zip_fread(zipFile, (void *)fileBuf.data(), fileBuf.size()) !=
      (zip_int64_t)fileBuf.size()) {
    auto *err = zip_file_get_error(zipFile);
    return extractLibZipError("Reading data from file within zip", *err);
  }

  if (auto errorCode = zip_fclose(zipFile) != 0) {
    zip_error_t err;

    zip_error_init_with_code(&err, errorCode);
    return extractLibZipError("Closing file in zip", err);
  }

  return fileBuf;
}

llvm::Error parseSignature(zip_t *zip, Signature &sig,
                           const std::shared_ptr<PatchableBinary> &binary) {
  zip_stat_t zs;
  auto numEntries = zip_get_num_entries(zip, 0);

  assert(numEntries >= 0);
  zip_stat_init(&zs);

  for (ssize_t i = 0; i < numEntries; i++) {
    zip_stat_index(zip, i, 0, &zs);

    llvm::StringRef name(zs.name);

    if (!name.endswith(".parmmap"))
      continue;

    auto fileBufOrErr = readFileFromZip(zip, zs);

    if (!fileBufOrErr)
      return fileBufOrErr.takeError();

    auto &fileBuf = fileBufOrErr.get();

    binary->parseParamMapIntoSignature(fileBuf, name, sig);
  }

  return llvm::Error::success();
}

llvm::Expected<Signature>
parseSignature(zip_t *zip, const std::shared_ptr<PatchableBinary> &binary) {
  Signature sig;

  if (auto err = parseSignature(zip, sig, binary))
    return std::move(err);

  return sig;
}

llvm::Error updateParameters(qssc::payload::PatchableZipPayload &payload,
                             Signature &sig, ArgumentSource const &arguments,
                             PatchableBinaryFactory *binaryFactory) {

  for (auto &entry : sig.patchPointsByBinary) {
    auto binaryName = entry.getKey();
    auto patchPoints = entry.getValue();

    if (patchPoints.size() == 0) // no patch points
      continue;

    auto binaryDataOrErr = payload.readMember(binaryName);

    if (!binaryDataOrErr)
      return binaryDataOrErr.takeError();

    auto &binaryData = binaryDataOrErr.get();

    auto binary =
        std::shared_ptr<PatchableBinary>(binaryFactory->create(binaryData));

    for (auto const &patchPoint : entry.getValue())
      if (auto err = binary->patch(patchPoint, arguments))
        return err;
  }

  return llvm::Error::success();
}

llvm::Error bindArguments(llvm::StringRef moduleInputPath,
                          llvm::StringRef payloadOutputPath,
                          ArgumentSource const &arguments,
                          PatchableBinaryFactory *factory) {

  std::error_code copyError =
      llvm::sys::fs::copy_file(moduleInputPath, payloadOutputPath);

  if (copyError)
    return llvm::make_error<llvm::StringError>(
        "Failed to copy circuit module to payload", copyError);

  qssc::payload::PatchableZipPayload payload(payloadOutputPath);

  Signature sig;

  auto binary = std::shared_ptr<PatchableBinary>(factory->create());
  if (auto err = parseSignature(payload.getBackingZip(), sig, binary))
    return err;

  if (auto err = updateParameters(payload, sig, arguments, factory))
    return err;

  if (auto err = payload.writeBack())
    return err;

  return llvm::Error::success();
}

} // namespace qssc::parameters
