//===- qss-compiler.cpp -----------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#include "API/api.h"

auto main(int argc, const char **argv) -> int {
  return compile(argc, argv, nullptr);
}
