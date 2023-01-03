//===- QUIRVariableBuilder.h ------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file declares the generator for variable handling code
///  QUIRVariableBuilder.
///
//===----------------------------------------------------------------------===//

#ifndef OPENQASM3_QUIR_VARIABLE_BUILDER_H
#define OPENQASM3_QUIR_VARIABLE_BUILDER_H

#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>

#include <qasm/AST/ASTSymbolTable.h>
#include <qasm/AST/ASTTypes.h>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Optional.h>
#include <llvm/ADT/StringRef.h>

#include <unordered_map>

namespace qssc::frontend::openqasm3 {

class QUIRVariableBuilder {
public:
  QUIRVariableBuilder(mlir::OpBuilder &builder) : builder(builder) {}

  /// Generate code for declaring a variable (at the builder's current insertion
  /// point).
  ///
  /// @param location source location related to the generated code.
  /// @param variableName name of the variable.
  /// @param type type of the variable.
  /// @param isInputVariable whether the variable is an "input" OpenQASM3
  /// variable
  /// @param isOutputVariable whether the variable is an "output" OpenQASM3
  /// variable
  void generateVariableDeclaration(mlir::Location location,
                                   llvm::StringRef variableName,
                                   mlir::Type type,
                                   bool isInputVariable = false,
                                   bool isOutputVariable = false);

  /// Generate code for declaring an array (at the builder's current insertion
  /// point).
  ///
  /// @param location source location related to the generated code.
  /// @param variableName name of the array variable.
  /// @param elementType element type of the array.
  /// @param width number of elements in the array.
  void generateArrayVariableDeclaration(mlir::Location location,
                                        llvm::StringRef variableName,
                                        mlir::Type elementType, int64_t width);

  /// Generate code for using a variable's current value (at the builder's
  /// current insertion point).
  ///
  /// @param location source location related to the generated code.
  /// @param variableName name of the variable.
  /// @param variableType type of the variable.
  ///
  /// @returns mlir::Value that delivers the variable's live value at that point
  /// in the code.
  mlir::Value generateVariableUse(mlir::Location location,
                                  llvm::StringRef variableName,
                                  mlir::Type variableType);

  /// Generate code for using a variable's current value (at the builder's
  /// current insertion point).
  ///
  /// @param location source location related to the generated code.
  /// @param variableName name of the variable.
  /// @param symTabEntry symbol table entry that describes the variable.
  ///
  /// @returns mlir::Value that delivers the variable's live value at that point
  /// in the code.
  mlir::Value
  generateVariableUse(mlir::Location location, llvm::StringRef variableName,
                      QASM::ASTSymbolTableEntry const *symTabEntry) {
    assert(symTabEntry);
    return generateVariableUse(location, variableName,
                               resolveQUIRVariableType(symTabEntry));
  }

  /// Generate code for using a variable's current value (at the builder's
  /// current insertion point).
  ///
  /// @param location source location related to the generated code.
  /// @param id identifier node that refers to the variable (must contain link
  /// to a symbol table entry)
  ///
  /// @returns mlir::Value that delivers the variable's live value at that point
  /// in the code.
  mlir::Value generateVariableUse(mlir::Location location,
                                  const QASM::ASTIdentifierNode *id) {
    assert(id->HasSymbolTableEntry());
    return generateVariableUse(
        location, id->GetName(),
        resolveQUIRVariableType(id->GetSymbolTableEntry()));
  }

  /// Generate code for using an element in an array variable (at the builder's
  /// current insertion point).
  ///
  /// @param location source location related to the generated code.
  /// @param variableName name of the array variable.
  /// @param elementIndex index of the array element to access.
  /// @param elementType type of the array's elements.
  ///
  /// @returns mlir::Value that delivers the array element's live value at that
  /// point in the code.
  mlir::Value generateArrayVariableElementUse(mlir::Location location,
                                              llvm::StringRef variableName,
                                              size_t elementIndex,
                                              mlir::Type elementType);

  /// Generate code for using an element in an array variable (at the builder's
  /// current insertion point).
  ///
  /// @param location source location related to the generated code.
  /// @param variableName name of the array variable.
  /// @param elementIndex index of the array element to access.
  /// @param symTabEntry symbol table entry that describes the array.
  ///
  /// @returns mlir::Value that delivers the array element's live value at that
  /// point in the code.
  mlir::Value generateArrayVariableElementUse(
      mlir::Location location, llvm::StringRef variableName,
      size_t elementIndex, QASM::ASTSymbolTableEntry const *symTabEntry) {
    assert(symTabEntry);
    return generateArrayVariableElementUse(
        location, variableName, elementIndex,
        resolveQUIRVariableType(symTabEntry));
  }

  /// Generate code for assigning a new value to a variable (at the builder's
  /// current insertion point).
  ///
  /// @param location source location related to the generated code.
  /// @param variableName name of the variable.
  /// @param assignedValue the value to assign to the variable.
  void generateVariableAssignment(mlir::Location location,
                                  llvm::StringRef variableName,
                                  mlir::Value assignedValue);

  /// Generate code for assigning a new value to an element in an array (at the
  /// builder's current insertion point).
  ///
  /// @param location source location related to the generated code.
  /// @param variableName name of the variable.
  /// @param assignedValue the value to assign to the array element.
  /// @param elementIndex index of the array element to assign to.
  void generateArrayVariableElementAssignment(mlir::Location location,
                                              llvm::StringRef variableName,
                                              mlir::Value assignedValue,
                                              size_t elementIndex);

  /// Generate code for assigning a single bit in a classical bit register (at
  /// the builder's current insertion point).
  ///
  /// @param location source location related to the generated code.
  /// @param variableName name of the cbit register.
  /// @param assignedValue the bit to assign to the array element.
  /// @param bitPosition the position to assign in the cbit register
  /// @param registerWidth the width of the input cbit register
  void generateCBitSingleBitAssignment(mlir::Location location,
                                       llvm::StringRef variableName,
                                       mlir::Value assignedValue,
                                       size_t bitPosition,
                                       size_t registerWidth);

  /// Return whether there is a variable with the given name tracked by this
  /// variable handler. This function is a cludge while transitioning
  /// variable handling.
  bool tracksVariable(llvm::StringRef variableName) {

    return variables.find(variableName.str()) != variables.end();
  }

  /// Resolve the mlir::Type for representing a given symbol table entry.
  mlir::Type
  resolveQUIRVariableType(QASM::ASTSymbolTableEntry const *entry) const;

  mlir::Type resolveQUIRVariableType(const QASM::ASTDeclarationNode *node);

  mlir::Type resolveQUIRVariableType(const QASM::ASTResultNode *node);

private:
  mlir::OpBuilder &builder;

  std::unordered_map<std::string, mlir::Type> variables;

  std::unordered_map<mlir::Operation *, mlir::Operation *> lastDeclaration;

  mlir::Type resolveQUIRVariableType(QASM::ASTType astType,
                                     const unsigned bits) const;
};

} // namespace qssc::frontend::openqasm3

#endif // OPENQASM3_QUIR_VARIABLE_BUILDER_H
