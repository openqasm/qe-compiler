//===- Arguments.h ---------------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file declares an top level interface for patching circuit arguments
///  after compilation. These are abstract class. A target will need to define
///  concrete classes to utilize the bindArguments interface.
///
//===----------------------------------------------------------------------===//

#ifndef ARGUMENTS_H
#define ARGUMENTS_H

#include "API/errors.h"
#include "Dialect/QCS/IR/QCSTypes.h"

#include "Arguments/Signature.h"
#include "Payload/Payload.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace qssc::arguments {

using ArgumentType = std::variant<std::optional<double>>;

class ArgumentSource {
public:
  virtual ArgumentType getArgumentValue(llvm::StringRef name) const = 0;

  virtual ~ArgumentSource() = default;
};

// BindArgumentsImplementation - abstract class to be subclassed by targets to
// define and implement methods for binding arguments to compiled payloads
class BindArgumentsImplementation {
public:
  virtual ~BindArgumentsImplementation() = default;
  virtual llvm::Error patch(PatchPoint const &patchPoint,
                            ArgumentSource const &arguments) = 0;
  virtual llvm::Error
  parseParamMapIntoSignature(llvm::StringRef paramMapContents,
                             llvm::StringRef paramMapFileName,
                             qssc::arguments::Signature &sig) = 0;
  virtual qssc::payload::PatchablePayload *
  getPayload(llvm::StringRef payloadOutputPath, bool enableInMemory) = 0;
  virtual llvm::Expected<Signature>
  parseSignature(qssc::payload::PatchablePayload *payload) = 0;
  void setTreatWarningsAsErrors(bool val) { treatWarningsAsErrors_ = val; }

protected:
  bool treatWarningsAsErrors_{false};
};

// BindArgumentsImplementationFactory - abstract class to be subclassed by
// targets that define and implement a factory for creating
// BindArgumentsImplementation objects
class BindArgumentsImplementationFactory {
public:
  virtual ~BindArgumentsImplementationFactory() = default;
  virtual BindArgumentsImplementation *
  create(OptDiagnosticCallback onDiagnostic) = 0;
  virtual BindArgumentsImplementation *
  create(std::vector<char> &buf, OptDiagnosticCallback onDiagnostic) = 0;
  virtual BindArgumentsImplementation *
  create(std::string &str, OptDiagnosticCallback onDiagnostic) = 0;
};

// TODO generalize type of arguments
llvm::Error bindArguments(llvm::StringRef moduleInput,
                          llvm::StringRef payloadOutputPath,
                          ArgumentSource const &arguments,
                          bool treatWarningsAsErrors, bool enableInMemoryInput,
                          std::string *inMemoryOutput,
                          BindArgumentsImplementationFactory &factory,
                          const OptDiagnosticCallback &onDiagnostic);

} // namespace qssc::arguments

#endif // ARGUMENTS_H
