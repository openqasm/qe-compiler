//===- SystemConfiguration.h - Config info ----------------------*- C++ -*-===//
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
//
//  This file declares the classes for config data
//
//===----------------------------------------------------------------------===//
#ifndef QSSC_SYSTEMCONFIGURATION_H
#define QSSC_SYSTEMCONFIGURATION_H

#include "Payload/Payload.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/None.h"

#include <istream>
#include <memory>

namespace qssc::hal {

class SystemConfiguration {
public:
  uint getNumQubits() const { return numQubits; }

  virtual ~SystemConfiguration();

protected:
  SystemConfiguration() = default;

  uint numQubits;
};
} // namespace qssc::hal
#endif // QSSC_SYSTEMCONFIGURATION_H
