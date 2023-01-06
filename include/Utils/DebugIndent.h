//===- DebugIndent.h  - Mixin class for debug indentation -------*- C++ -*-===//
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
  void increaseDebugIndent();
  void decreaseDebugIndent();

private:
  unsigned int debugIndentCount{0};
  unsigned int debugIndentStep{2};
};

} // namespace qssc::utils

#endif // UTILS_DEBUG_INDENT_H
