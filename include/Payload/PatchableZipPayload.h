//===- PatchableZipPayload.h ------------------------------------*- C++ -*-===//
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
///  This file defines the interface for patching zip payloads
///
//===----------------------------------------------------------------------===//

#ifndef PAYLOAD_PATCHABLE_ZIP_PAYLOAD_H
#define PAYLOAD_PATCHABLE_ZIP_PAYLOAD_H

#include "Arguments/Arguments.h"
#include "Payload/Payload.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <string>
#include <unordered_map>
#include <zip.h>

namespace qssc::payload {
class PatchableZipPayload : public PatchablePayload {
public:
  PatchableZipPayload(std::string path, bool enableInMemory)
      : path(std::move(path)), zip(nullptr), enableInMemory(enableInMemory) {}
  PatchableZipPayload(llvm::StringRef path, bool enableInMemory)
      : path(path), zip(nullptr), enableInMemory(enableInMemory) {}

  // deny copying and moving (no need for special handling of the resource
  // struct zip *)
  PatchableZipPayload(const PatchableZipPayload &) = delete;
  PatchableZipPayload &operator=(const PatchableZipPayload &) = delete;
  PatchableZipPayload(PatchableZipPayload &&) = delete;
  PatchableZipPayload &operator=(PatchableZipPayload &&) = delete;

  ~PatchableZipPayload();

  llvm::Error writeBack() override;
  llvm::Error writeString(std::string *outputString) override;
  void discardChanges();

  using ContentBuffer = std::vector<char>;

  llvm::Expected<ContentBuffer &>
  readMember(llvm::StringRef path, bool markForWriteBack = true) override;

  struct zip *getBackingZip() {
    if (auto err = ensureOpen()) {
      llvm::errs() << err;
      return nullptr;
    }

    return zip;
  }

private:
  struct TrackedFile {
    bool writeBack;
    ContentBuffer buf;
  };

  std::string const path;
  struct zip *zip;
  bool enableInMemory;

  std::unordered_map<std::string, TrackedFile> files;

  llvm::Error ensureOpen();
  llvm::Error addFileToZip(zip_t *zip, const std::string &path,
                           ContentBuffer &buf, zip_error_t &err);
};

llvm::Error extractLibZipError(llvm::StringRef info, zip_error_t &zipError);
llvm::Expected<std::string> readFileFromZip(zip_t *zip, zip_stat_t &zs);

} // namespace qssc::payload

#endif // PAYLOAD_PATCHABLE_ZIP_PAYLOAD_H
