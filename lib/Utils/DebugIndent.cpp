//===- DebugIndent.cpp - Mixin class for debug indentation ------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
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

#include "Utils/DebugIndent.h"

#include <string>

using namespace qssc::utils;

DebugIndent::DebugIndent(unsigned int indentStep)
    : debugIndentStep(indentStep) {}

// DEBUG versions of the {in,de}creaseDebugIndent methods are defined here
// NDEBUG versions are defined in the header file as empty methods

#ifndef NDEBUG
void DebugIndent::increaseDebugIndent() { debugIndentCount += debugIndentStep; }
#endif

#ifndef NDEBUG
void DebugIndent::decreaseDebugIndent() {
  if (debugIndentCount >= debugIndentStep)
    debugIndentCount -= debugIndentStep;
  else
    debugIndentCount = 0;
}
#endif

std::string DebugIndent::indent() {
  // this method is intended to be called inside LLVM_DEBUG
  // and therefore does not require a NDEBUG test
  return {std::string(debugIndentCount, ' ')};
}
