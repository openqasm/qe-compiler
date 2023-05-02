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

#include "llvm/Support/Error.h"

#include <zip.h>

#include <unordered_map>

namespace qssc::payload {
class PatchableZipPayload {
public:
  PatchableZipPayload(std::string path) : path(std::move(path)), zip(nullptr) {}
  PatchableZipPayload(llvm::StringRef path) : path(path), zip(nullptr) {}

  // deny copying and moving (no need for special handling of the resource
  // struct zip *)
  PatchableZipPayload(const PatchableZipPayload &) = delete;
  PatchableZipPayload &operator=(const PatchableZipPayload &) = delete;
  PatchableZipPayload(PatchableZipPayload &&) = delete;
  PatchableZipPayload &operator=(PatchableZipPayload &&) = delete;

  ~PatchableZipPayload();

  llvm::Error writeBack();
  void discardChanges();

  using ContentBuffer = std::vector<char>;

  llvm::Expected<ContentBuffer &> readMember(llvm::StringRef path,
                                             bool markForWriteBack = true);

  struct zip *getBackingZip() {    // TODO remove after cleanup
    if (auto err = ensureOpen()) { 
      llvm::errs() << err;
      return nullptr;
    }

    return zip;
  } // TODO remove after cleanup

private:
  struct TrackedFile {
    bool writeBack;
    ContentBuffer buf;
  };

  std::string const path;
  struct zip *zip;

  std::unordered_map<std::string, TrackedFile> files;

  llvm::Error ensureOpen();
};

llvm::Error extractLibZipError(llvm::StringRef info,
                                      zip_error_t &zipError);

} // namespace qssc::payload

#endif // PAYLOAD_PATCHABLE_ZIP_PAYLOAD_H