//===- qss-compiler.cpp -----------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#include "API/api.h"

int main(int argc, const char **argv) {
  return qssc::compile(argc, argv, nullptr, {});
}
