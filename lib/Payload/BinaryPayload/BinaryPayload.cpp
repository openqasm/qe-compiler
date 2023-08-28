//===- BinaryPayload.cpp ----------------------------------------*- C++ -*-===//
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
/// Implements the BinaryPayload class
///
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <ostream>
#include <sys/stat.h>
#include <unordered_set>

#include "BinaryPayload.h"
#include "Config.h"

#include "Payload/PayloadRegistry.h"

using namespace qssc::payload;
namespace fs = std::filesystem;

int qssc::payload::init_binary_payload() {
  const char *name = "BINARY";
  bool registered = registry::PayloadRegistry::registerPlugin(
      name, name, "Payload that generates a single binary for simulator",
      [](llvm::Optional<PayloadConfig> config)
          -> llvm::Expected<std::unique_ptr<payload::Payload>> {
        if (config.hasValue())
          return std::make_unique<BinaryPayload>(config.getValue());
        return std::make_unique<BinaryPayload>();
      });
  return registered ? 0 : -1;
}

void BinaryPayload::writePlain(llvm::raw_ostream &stream) {
  std::vector<fs::path> orderedNames = orderedFileNames();
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

void BinaryPayload::writePlain(std::ostream &stream) {
  llvm::raw_os_ostream llstream(stream);
  writePlain(llstream);
}

void BinaryPayload::write(llvm::raw_ostream &stream) {
  // single binary data is supported
  assert(orderedFileNames().size() == 1);

  for (auto &&filename : orderedFileNames()) {
    // write binary data to stream
    stream << files[filename];
  }
}

void BinaryPayload::write(std::ostream &stream) {
  // single binary data is supported
  assert(orderedFileNames().size() == 1);

  for (auto &&filename : orderedFileNames()) {
    // write binary data to stream
    stream << files[filename];
  }
}
