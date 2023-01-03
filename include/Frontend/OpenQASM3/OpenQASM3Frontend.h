//===- OpenQASM3Frontend.h --------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the interface to the OpenQASM 3 frontend.
///
//===----------------------------------------------------------------------===//

#ifndef OPENQASM3_FRONTEND_H
#define OPENQASM3_FRONTEND_H

#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/Error.h"

namespace qssc::frontend::openqasm3 {

llvm::Error parseOpenQASM3(std::string const &source, bool sourceIsFilename,
                           llvm::ArrayRef<std::string> includeDirs,
                           bool emitRawAST, bool emitPrettyAST, bool emitMLIR,
                           mlir::ModuleOp &newModule, unsigned int numShots,
                           std::string const &shotDelay);

}; // namespace qssc::frontend::openqasm3

#endif // OPENQASM3_FRONTEND_H
