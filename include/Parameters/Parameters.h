//===- Parameters.h ---------------------------------------------*- C++ -*-===//
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
///  This file declares parameter binding interface for targets to subclass.
///
//===----------------------------------------------------------------------===//

#ifndef PARAMETERS_H
#define PARAMETERS_H

#include "Parameters/Signature.h"
#include "Payload/Payload.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace qssc::parameters {

class ParameterSource {
public:
  virtual double getAngleParameter(llvm::StringRef name) const = 0;

  virtual ~ParameterSource() = default;
};

// PatchableBinary - abstract class to be subclassed by targets to
// define and implement patchable target binaries
class PatchableBinary {
public:
  virtual ~PatchableBinary() = default;
  virtual llvm::Error patch(PatchPoint const &patchPoint,
                            ParameterSource const &parameters) = 0;
  virtual void parseParamMapIntoSignature(llvm::StringRef paramMapContents,
                                          llvm::StringRef paramMapFileName,
                                          qssc::parameters::Signature &sig) = 0;
};

// PatchableBinary - abstract class to be subclassed by targets to
// define and implement a factory for creating PatchableBinary objects
class PatchableBinaryFactory {
public:
  virtual ~PatchableBinaryFactory() = default;
  virtual PatchableBinary *create() = 0;
  virtual PatchableBinary *create(std::vector<char> &buf) = 0;
  virtual PatchableBinary *create(std::string &str) = 0;
};

// NullPatchableBinary - concrete implementation of PatchableBinary for
// targets that do not want to support patchable binaries
class NullPatchableBinary : public PatchableBinary {
  llvm::Error patch(PatchPoint const &patchPoint,
                    ParameterSource const &parameters) override {
    return llvm::Error::success();
  };
  void parseParamMapIntoSignature(llvm::StringRef paramMapContents,
                                  llvm::StringRef paramMapFileName,
                                  qssc::parameters::Signature &sig) override{};
};

// NullPatchableBinary - concrete implementation of PatchableBinaryFactory for
// targets that do not want to support patchable binaries
class NullPatchableBinaryFactory : public PatchableBinaryFactory {
  PatchableBinary *create() override { return new NullPatchableBinary(); }
  PatchableBinary *create(std::vector<char> &buf) override { return create(); }
  PatchableBinary *create(std::string &str) override { return create(); }
};

// TODO generalize type of parameters
llvm::Error bindParameters(llvm::StringRef moduleInputPath,
                           llvm::StringRef payloadOutputPath,
                           ParameterSource const &parameters,
                           PatchableBinaryFactory *factory);

} // namespace qssc::parameters

#endif // PARAMETERS_H
