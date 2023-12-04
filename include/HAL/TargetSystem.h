//===- TargetSystem.h - Top-level target info -------------------*- C++ -*-===//
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
//  This file declares the classes for the top-level target interface
//
//===----------------------------------------------------------------------===//
#ifndef TARGETSYSTEM_H
#define TARGETSYSTEM_H

#include "Arguments/Arguments.h"

#include "llvm/ADT/Optional.h"
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
public:
  using PluginConfiguration = llvm::StringRef;

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

  virtual llvm::Error addPayloadPasses(mlir::PassManager &pm,
                                       bool generatePayload = false) = 0;
  virtual llvm::Error addToPayload(mlir::ModuleOp &moduleOp,
                                   payload::Payload &payload) = 0;

  virtual llvm::Optional<qssc::arguments::BindArgumentsImplementationFactory *>
  getBindArgumentsImplementationFactory() {
    return llvm::None;
  };

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
