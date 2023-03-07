//===- DebugIndent.cpp - Mixin class for debug indentation ------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
