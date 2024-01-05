//===- Payload.h ------------------------------------------------*- C++ -*-===//
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
//
// Declares the Payload wrapper class
//
//===----------------------------------------------------------------------===//

#ifndef PAYLOAD_PAYLOAD_H
#define PAYLOAD_PAYLOAD_H

#include <Config/QSSConfig.h>

#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

namespace qssc::payload {

struct PayloadConfig {
  std::string prefix;
  std::string name;
  qssc::config::QSSVerbosity verbosity;
};

// Payload class will wrap the QSS Payload and interface with the qss-compiler
class Payload {
public:
  using PluginConfiguration = PayloadConfig;

public:
  Payload()
      : prefix(""), name("exp"), verbosity(qssc::config::QSSVerbosity::Warn) {}
  explicit Payload(PayloadConfig config)
      : prefix(std::move(config.prefix) + "/"), name(std::move(config.name)),
        verbosity(config.verbosity) {
    files.clear();
  }
  virtual ~Payload() = default;

  // get/add the file fName and return a pointer to its data
  std::string *getFile(const std::string &fName);
  // get/add the file fName and return a pointer to its data
  std::string *getFile(const char *fName);
  // write all files to the stream
  virtual void write(llvm::raw_ostream &stream) = 0;
  // write all files to the stream
  virtual void write(std::ostream &stream) = 0;
  // write all files in plaintext to the stream
  virtual void writePlain(std::ostream &stream) = 0;
  virtual void writePlain(llvm::raw_ostream &stream) = 0;
  virtual void addFile(llvm::StringRef filename, llvm::StringRef str) = 0;

  const std::string &getName() const { return name; }
  const std::string &getPrefix() const { return prefix; }

protected:
  // Class mutex
  std::mutex _mtx;

  // return an ordered list of filenames
  auto orderedFileNames() -> std::vector<std::filesystem::path>;

  // A hash function object to work with unordered_* containers:
  struct PathHash {
    std::size_t operator()(std::filesystem::path const &p) const noexcept {
      return std::filesystem::hash_value(p);
    }
  };

  std::string prefix;
  std::string name;
  qssc::config::QSSVerbosity verbosity;
  std::unordered_map<std::filesystem::path, std::string, PathHash> files;
}; // class Payload

// PatchablePayload for payloads that support patching after compilation
class PatchablePayload {
public:
  virtual ~PatchablePayload() = default;
  using ContentBuffer = std::vector<char>;
  virtual llvm::Expected<ContentBuffer &>
  readMember(llvm::StringRef path, bool markForWriteBack = true) = 0;
  virtual llvm::Error writeBack() = 0;
  virtual llvm::Error writeString(std::string *outputString) = 0;
}; // class PatchablePayload

} // namespace qssc::payload

#endif // PAYLOAD_PAYLOAD_H
