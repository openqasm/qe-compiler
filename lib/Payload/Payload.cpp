//===- Payload.cpp ----------------------------------------------*- C++ -*-===//
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
//
// Implements the Payload wrapper class
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <fstream>
#include <ostream>
#include <unordered_set>

#include "nlohmann/json.hpp"

#include "Payload/Payload.h"

// Inject static initialization headers from payloads. We need to include them
// in a translation unit that is not being optimized (removed) by the compiler.
#include "Payloads.inc"

using namespace qssc::payload;
namespace fs = std::filesystem;

auto Payload::getFile(const std::string &fName) -> std::string * {
  const std::lock_guard<std::mutex> lock(_mtx);
  const std::string key = prefix + fName;
  files.try_emplace(key);
  return &files[key];
}

auto Payload::getFile(const char *fName) -> std::string * {
  const std::lock_guard<std::mutex> lock(_mtx);
  const std::string key = prefix + fName;
  files.try_emplace(key);
  return &files[key];
}

auto Payload::orderedFileNames() -> std::vector<fs::path> {
  const std::lock_guard<std::mutex> lock(_mtx);
  std::vector<fs::path> ret;
  for (auto &filePair : files)
    ret.emplace_back(filePair.first);
  std::sort(ret.begin(), ret.end());
  return ret;
}
