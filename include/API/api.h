//===- api.h ----------------------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#ifndef QSS_COMPILER_LIB_H
#define QSS_COMPILER_LIB_H

#include "mlir/IR/Dialect.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <functional>
#include <string>
#include <unordered_map>

int compile(int argc, char const **argv, std::string *outputString);

llvm::Error
bindParameters(llvm::StringRef target, llvm::StringRef moduleInputPath,
               llvm::StringRef payloadOutputPath,
               std::unordered_map<std::string, double> const &parameters);

/// Register all qss-compiler dialects returning a dialect registry
mlir::DialectRegistry registerDialects();

#endif // QSS_COMPILER_LIB_H
