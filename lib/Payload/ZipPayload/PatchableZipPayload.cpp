//===- PatchableZipPayload.cpp - file supporting parameters ------* C++ -*-===//
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
///  This file implements utility functions for patching files in zip archives
///  with updated arguments
///
//===----------------------------------------------------------------------===//

#include "Payload/PatchableZipPayload.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <zip.h>

namespace qssc::payload {

llvm::Expected<std::string> readFileFromZip(zip_t *zip, zip_stat_t &zs) {
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

llvm::Error extractLibZipError(llvm::StringRef info, zip_error_t &zipError) {
  std::string errorMsg;
  llvm::raw_string_ostream errorMsgStream(errorMsg);

  errorMsgStream << info << " " << zip_error_strerror(&zipError);

  auto errorCode = static_cast<int>(std::errc::invalid_argument);

  if (zip_error_system_type(&zipError) == ZIP_ET_SYS)
    errorCode = zip_error_code_system(&zipError);

  zip_error_fini(&zipError);

  return llvm::make_error<llvm::StringError>(
      errorMsg, std::error_code{errorCode, std::system_category()});
}

llvm::Error PatchableZipPayload::ensureOpen() {
  if (zip) // already open
    return llvm::Error::success();

  int errorCode;
  zip_error_t zipError;

  zip_error_init(&zipError);

  if ((zip = zip_open(path.c_str(), 0, &errorCode)) == nullptr) {
    zip_error_set(&zipError, errorCode, errno);
    return extractLibZipError(
        "Failure while opening circuit module (zip) file ", zipError);
  }

  zip_error_fini(&zipError);
  return llvm::Error::success();
}

void PatchableZipPayload::discardChanges() {
  if (zip == nullptr)
    return;

  zip_discard(zip);
  zip = nullptr;
}

llvm::Error PatchableZipPayload::writeBack() {
  if (zip == nullptr) // no changes pending, thus no operation
    return llvm::Error::success();

  zip_error_t err;

  zip_error_init(&err);

  for (auto &item : files) {
    if (!item.second.writeBack)
      continue;

    auto &path = item.first;
    auto &buf = item.second.buf;

    zip_source_t *src =
        zip_source_buffer_create(buf.data(), buf.size(), 0, &err);

    if (src == nullptr)
      return extractLibZipError("Creating zip source from data buffer", err);

    if (zip_file_add(zip, path.c_str(), src, ZIP_FL_OVERWRITE) == 0) {
      auto *archiveErr = zip_get_error(zip);

      zip_source_free(src);
      return extractLibZipError("Adding or replacing file to zip", *archiveErr);
    }
  }

  zip_error_fini(&err);

  if (zip_close(zip)) {
    auto *err = zip_get_error(zip);
    return extractLibZipError("writing payload file", *err);
  }

  zip = nullptr;
  return llvm::Error::success();
}

llvm::Error PatchableZipPayload::writeString(std::string *outputString) {
  if (outputString == nullptr) // no output buffer
    return llvm::make_error<llvm::StringError>("outputString buffer is null",
                                               llvm::inconvertibleErrorCode());

  llvm::outs() << "Setup string stream"
               << "\n";

  llvm::Optional<llvm::raw_string_ostream> outStringStream;
  outStringStream.emplace(*outputString);
  llvm::raw_ostream *ostream = outStringStream.getPointer();

  // load all files in zip if required

  auto numEntries = zip_get_num_entries(zip, 0);
  zip_stat_t zs;
  assert(numEntries >= 0);
  zip_stat_init(&zs);
  for (ssize_t i = 0; i < numEntries; i++) {
    zip_stat_index(zip, i, 0, &zs);
    llvm::StringRef name(zs.name);
    auto binaryDataOrErr = readMember(name);

    if (!binaryDataOrErr)
      return binaryDataOrErr.takeError();
  }

  zip_source_t *new_zip_src;
  zip_t *new_zip;

  zip_error_t err;

  zip_error_init(&err);

  // open a zip source, buffer is allocated internally to libzip
  if ((new_zip_src = zip_source_buffer_create(nullptr, 0, 0, &err)) == nullptr)
    return extractLibZipError("Can't create zip source for new archive", err);

  zip_source_keep(new_zip_src);

  if ((new_zip = zip_open_from_source(new_zip_src, ZIP_TRUNCATE, &err)) ==
      nullptr) {
    zip_source_free(new_zip_src);
    return extractLibZipError(
        "Can't create/open an archive from the new archive source:", err);
  }

  llvm::outs() << "Adding files"
               << "\n";

  for (auto &item : files) {
    auto &path = item.first;
    auto &buf = item.second.buf;

    llvm::outs() << "Adding " << path << "\n";
    llvm::outs() << "   zip_source_buffer_create\n";
    zip_source_t *src =
        zip_source_buffer_create(buf.data(), buf.size(), 0, &err);

    if (src == nullptr)
      return extractLibZipError("Creating zip source from data buffer", err);

    llvm::outs() << "   zip_file_add\n";
    if (zip_file_add(new_zip, path.c_str(), src, ZIP_FL_OVERWRITE) < 0) {
      auto *archiveErr = zip_get_error(new_zip);
      zip_source_free(new_zip_src);
      return extractLibZipError("Adding or replacing file to zip", *archiveErr);
    }
    llvm::outs() << "   done\n";
  }

  llvm::outs() << "Closing in memory zip"
               << "\n";

  zip_error_fini(&err);

  if (zip_close(new_zip)) {
    auto *err = zip_get_error(new_zip);
    return extractLibZipError("Closing in memory zip", *err);
  }

  llvm::outs() << "Reopen zip"
               << "\n";

  //===---- Reopen for copying ----===//
  zip_source_open(new_zip_src);
  zip_source_seek(new_zip_src, 0, SEEK_END);
  zip_int64_t sz = zip_source_tell(new_zip_src);

  // allocate a new buffer to copy the archive into
  char *outbuffer = (char *)malloc(sz);
  if (!outbuffer) {
    zip_source_close(new_zip_src);
    return llvm::make_error<llvm::StringError>(
        "Unable to allocate output buffer for writing zip to stream",
        llvm::inconvertibleErrorCode());
  }

  // seek back to the beginning of the archive
  zip_source_seek(new_zip_src, 0, SEEK_SET);
  zip_source_read(new_zip_src, outbuffer, sz);
  zip_source_close(new_zip_src);

  llvm::outs() << "Write to stream"
               << "\n";

  // output the new archive to the stream
  ostream->write(outbuffer, sz);
  ostream->flush();
  free(outbuffer);
  llvm::outs() << "Write String Done"
               << "\n";
  return llvm::Error::success();
}

llvm::Expected<PatchableZipPayload::ContentBuffer &>
PatchableZipPayload::readMember(llvm::StringRef path, bool markForWriteBack) {

  auto pathStr = path.operator std::string();
  auto pos = files.find(pathStr);

  if (pos != files.end())
    return pos->second.buf;

  zip_stat_t zs;

  zip_stat_init(&zs);

  if (zip_stat(zip, pathStr.c_str(), ZIP_FL_ENC_UTF_8, &zs) == -1) {
    auto *err = zip_get_error(zip);

    return extractLibZipError("Opening file within zip", *err);
  }

  auto *zipFile = zip_fopen_index(zip, zs.index, 0);

  if (!zipFile) {
    auto *err = zip_get_error(zip);
    return extractLibZipError("Opening file within zip", *err);
  }

  ContentBuffer fileBuf(zs.size, 0);

  if (zip_fread(zipFile, (void *)fileBuf.data(), fileBuf.size()) !=
      (zip_int64_t)fileBuf.size()) {
    auto *err = zip_file_get_error(zipFile);

    zip_fclose(zipFile);

    return extractLibZipError("Reading data from file within zip", *err);
  }

  if (auto errorCode = zip_fclose(zipFile) != 0) {
    zip_error_t err;

    zip_error_init_with_code(&err, errorCode);
    return extractLibZipError("Closing file in zip", err);
  }

  auto ins = files.emplace(std::make_pair(
      pathStr, TrackedFile{markForWriteBack, std::move(fileBuf)}));

  assert(ins.second && "expect insertion, i.e., had not been present before.");

  return ins.first->second.buf;
}

PatchableZipPayload::~PatchableZipPayload() {
  // discard any leftover changes that have not been written back
  if (zip)
    zip_discard(zip);
}

} // namespace qssc::payload
