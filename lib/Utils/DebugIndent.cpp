//===- DebugIndent.cpp - Mixin class for debug indentation ------*- C++ -*-===//
//
// (C) Copyright IBM 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file implements a base class for supporting indentation when debugging
///  passes using LLVM_DEBUG
///
///  This class is not intended to be instantiated as a concrete class but
///  rather used as a mixin for subclasses.
///
//===----------------------------------------------------------------------===//

#include <string>

#include "Utils/DebugIndent.h"

using namespace qssc::utils;

DebugIndent::DebugIndent(unsigned int indentStep)
    : debugIndentStep(indentStep) {}

void DebugIndent::increaseDebugIndent() {
#ifndef NDEBUG
  debugIndentCount += debugIndentStep;
#endif
}

void DebugIndent::decreaseDebugIndent() {
#ifndef NDEBUG
  if (debugIndentCount >= debugIndentStep)
    debugIndentCount -= debugIndentStep;
  else
    debugIndentCount = 0;
#endif
}

std::string DebugIndent::indent() {
  // this method is intended to be called inside LLVM_DEBUG
  // and therefore does not require a NDEBUG test
  return {std::string(debugIndentCount, ' ')};
}
