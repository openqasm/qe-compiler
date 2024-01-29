//===- qss-compiler.cpp -----------------------------------------*- C++ -*-===//
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

#include "API/api.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <utility>

int main(int argc, const char **argv) {

  auto err = qssc::compileMain(
      argc, argv, "Quantum System Software (QSS) Backend Compiler\n", {});
  if (err) {
    llvm::logAllUnhandledErrors(std::move(err), llvm::errs(), "Error: ");
    return EXIT_FAILURE;
  }
  return qssc::asMainReturnCode(std::move(err));
}
