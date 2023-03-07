//===- TargetSystem.h - Top-level target info -------------------*- C++ -*-===//
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
//  This file declares the classes for the top-level target interface
//
//===----------------------------------------------------------------------===//
#ifndef TARGETSYSTEM_H
#define TARGETSYSTEM_H

#include "llvm/ADT/SmallString.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include <string>
#include <vector>

namespace qssc::payload {
class Payload;
} // namespace qssc::payload

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace qssc::hal {
class Target;
class TargetSystem;
class TargetInstrument;

class Target {
protected:
  Target(std::string name, Target *parent);

public:
  virtual const std::string &getName() const { return name; }
  virtual llvm::StringRef getDescription() const { return ""; }
  Target *getParent() { return parent; }
  const Target *getParent() const { return parent; }
  virtual ~Target() = default;

protected:
  std::string name;

  // parent is already owned by unique_ptr
  // owned by the constructing context
  // class Target
  Target *parent;
};

class TargetSystem : public Target {
protected: // Can only create subclasses.
  TargetSystem(std::string name, Target *parent);

public:
  virtual std::vector<std::unique_ptr<TargetSystem>> &getChildren() {
    return children;
  }
  virtual const std::vector<std::unique_ptr<TargetSystem>> &
  getChildren() const {
    return children;
  }
  virtual std::vector<std::unique_ptr<TargetInstrument>> &getInstruments() {
    return instruments;
  }
  virtual const std::vector<std::unique_ptr<TargetInstrument>> &
  getInstruments() const {
    return instruments;
  }

  virtual llvm::Error addPayloadPasses(mlir::PassManager &pm) = 0;
  virtual llvm::Error addToPayload(mlir::ModuleOp &moduleOp,
                                   payload::Payload &payload) = 0;

  virtual ~TargetSystem() = default;

protected:
  std::vector<std::unique_ptr<TargetSystem>> children;
  std::vector<std::unique_ptr<TargetInstrument>> instruments;
}; // class TargetSystem

class TargetInstrument : public Target {
protected: // Can only create subclasses
  TargetInstrument(std::string name, Target *parent);

public:
  virtual auto addPayloadPasses(mlir::PassManager &pm) -> llvm::Error = 0;
  virtual auto addToPayload(mlir::ModuleOp &moduleOp, payload::Payload &payload)
      -> llvm::Error = 0;

  virtual ~TargetInstrument() = default;
}; // class TargetInstrument
} // namespace qssc::hal
#endif // TARGETSYSTEM_H
