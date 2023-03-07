//===- DebugIndent.h  - Mixin class for debug indentation -------*- C++ -*-===//
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
///  passes using LLVM_DEBUG.
///
///  This class is not intended to be instantiated as a concrete class but
///  rather used as a mixin for subclasses.
///
///  This class may be used by subclassing DebugIndent, #defining a unique
///  DEBUG_TYPE string for the class and then using the {in,de}creaseIndent()
///  methods and the INDENT_DEBUG and INDENT_DUMP macros.
///
//===----------------------------------------------------------------------===//

#ifndef UTILS_DEBUG_INDENT_H
#define UTILS_DEBUG_INDENT_H

#include <string>

// silence lint error for msg not in () - adding () will break macro
// NOLINTNEXTLINE(bugprone-macro-parentheses)
#define INDENT_DEBUG(msg) LLVM_DEBUG(llvm::errs() << indent() << msg)
#define INDENT_DUMP(msg) LLVM_DEBUG(llvm::errs() << indent(); (msg))

namespace qssc::utils {

class DebugIndent {
public:
  DebugIndent() = default;
  DebugIndent(unsigned int indentStep);

protected:
  std::string indent();

#ifdef NDEBUG
  void increaseDebugIndent() {}
  void decreaseDebugIndent() {}
#else
  void increaseDebugIndent();
  void decreaseDebugIndent();
#endif

private:
  unsigned int debugIndentCount{0};
  unsigned int debugIndentStep{2};
};

} // namespace qssc::utils

#endif // UTILS_DEBUG_INDENT_H
