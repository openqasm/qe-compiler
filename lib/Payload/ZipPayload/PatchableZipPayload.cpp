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
#include "ZipUtil.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <fstream>
#include <string_view>
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

  llvm::Error retVal = llvm::Error::success();

  int errorCode = 0;
  zip_error_t zipError;

  zip_error_init(&zipError);

  zip_source_t *zs;
  if (enableInMemory) {
    if ((zs = zip_source_buffer_create(path.data(), path.length(), 0,
                                       &zipError)) == nullptr) {
      zip_error_set(&zipError, errorCode, errno);
      retVal = extractLibZipError(
          "Failure while opening in memory circuit module (zip) ", zipError);
    }

    if ((zip = zip_open_from_source(zs, 0, &zipError)) == nullptr) {
      zip_error_set(&zipError, errorCode, errno);
      retVal = extractLibZipError(
          "Failure while opening in memory circuit module (zip) ", zipError);
    }
    inMemoryZipSource = zs;
  } else {
    if ((zip = zip_open(path.c_str(), 0, &errorCode)) == nullptr) {
      zip_error_set(&zipError, errorCode, errno);
      retVal = extractLibZipError(
          "Failure while opening circuit module (zip) file ", zipError);
    }
    inMemoryZipSource = nullptr;
  }

  zip_error_fini(&zipError);
  return retVal;
}

void PatchableZipPayload::discardChanges() {
  if (zip == nullptr)
    return;

  zip_discard(zip);
  zip = nullptr;
}

llvm::Error PatchableZipPayload::addFileToZip(zip_t *zip,
                                              const std::string &path,
                                              ContentBuffer &buf,
                                              zip_error_t &err) {

  zip_source_t *src = zip_source_buffer_create(buf.data(), buf.size(), 0, &err);

  if (src == nullptr)
    return extractLibZipError("Creating zip source from data buffer", err);

  if (zip_file_add(zip, path.c_str(), src, ZIP_FL_OVERWRITE) < 0) {
    auto *archiveErr = zip_get_error(zip);
    return extractLibZipError("Adding or replacing file to zip", *archiveErr);
  }
  return llvm::Error::success();
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
    auto error = addFileToZip(zip, path, buf, err);
    if (error)
      return error;
  }

  zip_error_fini(&err);

  if (inMemoryZipSource)
    zip_source_keep(inMemoryZipSource);

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

  llvm::Optional<llvm::raw_string_ostream> outStringStream;
  outStringStream.emplace(*outputString);
  llvm::raw_ostream *ostream = outStringStream.getPointer();

  if (inMemoryZipSource) {
    // read from in memory source
    zip_int64_t sz;
    char *outbuffer =
        qssc::payload::read_zip_src_to_buffer(inMemoryZipSource, sz);
    if (outbuffer) {
      ostream->write(outbuffer, sz);
      free(outbuffer);
    }
    zip_source_free(inMemoryZipSource);
    inMemoryZipSource = nullptr;
  } else {
    // re-read file from disk
    std::ostringstream buf;
    std::ifstream input(path.c_str());
    buf << input.rdbuf();
    ostream->write(buf.str().c_str(), buf.str().length());
  }
  ostream->flush();
  return llvm::Error::success();
}

llvm::Expected<PatchableZipPayload::ContentBuffer &>
PatchableZipPayload::readMember(llvm::StringRef path, bool markForWriteBack) {

  auto pathStr = path.operator std::string();
  auto pos = files.find(pathStr);

  if (pos != files.end())
    return pos->second.buf;

  if (enableInMemory &&
      (zip_name_locate(zip, pathStr.c_str(), ZIP_FL_ENC_UTF_8) == -1)) {
    // in memory payload does not have leading directory so attempt to remove
    auto index = path.find("/") + 1;
    path = path.substr(index);
    pathStr = path.operator std::string();
  }

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
