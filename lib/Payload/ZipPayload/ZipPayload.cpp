//===- ZipPayload.cpp -------------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023, 2024.
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
/// Implements the ZipPayload class
///
//===----------------------------------------------------------------------===//

#include "ZipPayload.h"

#include "Payload/Payload.h"
#include "ZipUtil.h"

#include "Config.h"
#include "Payload/PayloadRegistry.h"
#include <Config/QSSConfig.h>

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <ostream>
#include <string>
// NOLINTNEXTLINE(misc-include-cleaner)
#include <sys/stat.h>
#include <vector>
#include <zip.h>
#include <zipconf.h>

using namespace qssc::payload;
namespace fs = std::filesystem;

int qssc::payload::init() {
  const char *name = "ZIP";
  bool const registered = registry::PayloadRegistry::registerPlugin(
      name, name, "Payload that generates zip file with .qem extension.",
      [](std::optional<PayloadConfig> config)
          -> llvm::Expected<std::unique_ptr<payload::Payload>> {
        if (config.has_value())
          return std::make_unique<ZipPayload>(config.value());
        return std::make_unique<ZipPayload>();
      });
  return registered ? 0 : -1;
}

// creates a manifest json file and adds it to the file map
void ZipPayload::addManifest() {
  std::lock_guard<std::mutex> const lock(_mtx);
  std::string const manifest_fname = "manifest/manifest.json";
  nlohmann::json manifest;
  manifest["version"] = QSSC_VERSION;
  manifest["contents_path"] = prefix;
  files[manifest_fname] = manifest.dump() + "\n";
}

void ZipPayload::addFile(llvm::StringRef filename, llvm::StringRef str) {
  std::lock_guard<std::mutex> const lock(_mtx);
  files[filename.str()] = str;
}

void ZipPayload::writePlain(const std::string &dirName) {
  std::lock_guard<std::mutex> const lock(_mtx);
  for (const auto &filePair : files) {
    fs::path fName(dirName);
    fName /= filePair.first;

    fs::create_directories(fName.parent_path());
    std::ofstream fStream(fName, std::ofstream::out);
    if (fStream.fail() || !fStream.good()) {
      llvm::errs() << "Unable to open output file " << fName << "\n";
      continue;
    }
    fStream << filePair.second;
    fStream.close();
  }
}

void ZipPayload::writePlain(llvm::raw_ostream &stream) {
  std::vector<fs::path> const orderedNames = orderedFileNames();
  stream << "------------------------------------------\n";
  stream << "Plaintext payload: " << prefix << "\n";
  stream << "------------------------------------------\n";
  stream << "Manifest:\n";
  for (auto &fName : orderedNames)
    stream << fName << "\n";
  stream << "------------------------------------------\n";
  for (auto &fName : orderedNames) {
    stream << "File: " << fName << "\n";
    stream << files[fName];
    if (*(files[fName].rbegin()) != '\n')
      stream << "\n";
    stream << "------------------------------------------\n";
  }
}

void ZipPayload::writePlain(std::ostream &stream) {
  llvm::raw_os_ostream llstream(stream);
  writePlain(llstream);
}

namespace {
void setFilePermissions(zip_int64_t fileIndex, fs::path &fName,
                        zip_t *new_archive) {
  zip_uint8_t opsys;
  zip_uint32_t attributes;
  zip_file_get_external_attributes(new_archive, fileIndex, 0, &opsys,
                                   &attributes);
  if (opsys == ZIP_OPSYS_UNIX) {
    zip_uint32_t mask = UINT32_MAX; // all 1s for negative mask
    // NOLINTNEXTLINE(misc-include-cleaner)
    mask ^= (S_IWGRP << 16); // turn off write for the group
    // NOLINTNEXTLINE(misc-include-cleaner)
    mask ^= (S_IWOTH << 16); // turn off write for others

    // apply negative write mask
    attributes &= mask;

    // if executable turn on S_IXUSR
    if (fName.has_extension() && fName.extension() == ".sh")
      // NOLINTNEXTLINE(misc-include-cleaner)
      attributes |= (S_IXUSR << 16); // turn on execute for user

    // set new attributes
    zip_file_set_external_attributes(new_archive, fileIndex, 0, opsys,
                                     attributes);
  }
}
} // end anonymous namespace

void ZipPayload::writeZip(llvm::raw_ostream &stream) {
  if (verbosity >= qssc::config::QSSVerbosity::Info)
    llvm::outs() << "Writing zip to stream\n";
  // first add the manifest
  addManifest();

  // zip archive stuff
  zip_source_t *new_archive_src;
  zip_source_t *file_src;
  zip_t *new_archive;
  zip_error_t error;

  //===---- Initialize archive ----===//
  zip_error_init(&error);

  // open a zip source, buffer is allocated internally to libzip
  new_archive_src = zip_source_buffer_create(nullptr, 0, 0, &error);
  if (new_archive_src == nullptr) {
    llvm::errs() << "Can't create zip source for new archive: "
                 << zip_error_strerror(&error) << "\n";
    zip_error_fini(&error);
    return;
  }

  // make sure the new source buffer stays around after closing the archive
  zip_source_keep(new_archive_src);

  // create and open an archive from the new archive source
  new_archive = zip_open_from_source(new_archive_src, ZIP_TRUNCATE, &error);
  if (new_archive == nullptr) {
    llvm::errs() << "Can't create/open an archive from the new archive source: "
                 << zip_error_strerror(&error) << "\n";
    zip_source_free(new_archive_src);
    zip_error_fini(&error);
    return;
  }
  zip_error_fini(&error);

  if (verbosity >= qssc::config::QSSVerbosity::Info)
    llvm::outs() << "Zip buffer created, adding files to archive\n";
  // archive is now allocated and created, need to fill it with files/data
  std::vector<fs::path> orderedNames = orderedFileNames();
  for (auto &fName : orderedNames) {
    if (verbosity >= qssc::config::QSSVerbosity::Info)
      llvm::outs() << "Adding file " << fName << " to archive buffer ("
                   << files[fName].size() << " bytes)\n";

    //===---- Add file ----===//
    // init the error object
    zip_error_init(&error);

    // first create a zip source from the file data
    file_src = zip_source_buffer_create(files[fName].c_str(),
                                        files[fName].size(), 0, &error);
    if (file_src == nullptr) {
      llvm::errs() << "Can't create zip source for " << fName << " : "
                   << zip_error_strerror(&error) << "\n";
      zip_error_fini(&error);
      continue;
    }
    zip_error_fini(&error);

    // now add it to the archive
    zip_int64_t const fileIndex =
        zip_file_add(new_archive, fName.c_str(), file_src, ZIP_FL_OVERWRITE);
    if (fileIndex < 0) {
      llvm::errs() << "Problem adding file " << fName
                   << " to archive: " << zip_strerror(new_archive) << "\n";
      continue;
    }
    zip_set_file_compression(new_archive, fileIndex, ZIP_CM_STORE, 1);

    setFilePermissions(fileIndex, fName, new_archive);
  }

  //===---- Shutdown archive ----===//
  // shutdown the archive, write central directory
  if (zip_close(new_archive) < 0) {
    llvm::errs() << "Problem closing new zip archive: "
                 << zip_strerror(new_archive) << "\n";
    return;
  }

  //===---- Reopen for copying ----===//
  zip_int64_t sz;
  char *outbuffer = read_zip_src_to_buffer(new_archive_src, sz);
  if (verbosity >= qssc::config::QSSVerbosity::Info)
    llvm::outs() << "Zip buffer is of size " << sz << " bytes\n";
  if (outbuffer) {
    // output the new archive to the stream
    stream.write(outbuffer, sz);
    stream.flush();
    free(outbuffer);
  }
}

void ZipPayload::writeZip(std::ostream &stream) {
  llvm::raw_os_ostream llstream(stream);
  writeZip(llstream);
}

void ZipPayload::write(llvm::raw_ostream &stream) { writeZip(stream); }

void ZipPayload::write(std::ostream &stream) { writeZip(stream); }
