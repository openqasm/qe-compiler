//===- SystemConfiguration.h - Config info ----------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2023.
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
