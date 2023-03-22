//===- LegacyInputConversion.h ----------------------------------*- C++ -*-===//
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

#ifndef UTILS_LEGACY_INPUT_CONVERSION_H
#define UTILS_LEGACY_INPUT_CONVERSION_H

#include "Utils/SystemDefinition.h"

#include "mlir/IR/Operation.h"

#include <string>

namespace qssc::utils {

class LegacyInputConversion : public SystemDefinition {

public:
  ~LegacyInputConversion() = default;
  LegacyInputConversion(mlir::Operation *op) {}
  void create(const std::string &calibrationsFilename,
              const std::string &expParamsFilename,
              const std::string &backendConfigFilename);
};

} // namespace qssc::utils

#endif // UTILS_LEGACY_INPUT_CONVERSION_H
