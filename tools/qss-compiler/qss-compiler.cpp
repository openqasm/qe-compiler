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
  qssc::DiagnosticCallback diagnosticDiscardingCallback{
      [](qssc::Diagnostic const &) {
        /* diagnostics not handled separately */
      }};

  return qssc::compile(argc, argv, nullptr, diagnosticDiscardingCallback);
}
