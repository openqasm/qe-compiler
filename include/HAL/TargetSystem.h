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

#include "llvm/ADT/iterator_range.h"
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

  virtual const std::vector<std::unique_ptr<Target>> &getChildren_() const = 0;

public:
  virtual const std::string &getName() const { return name; }
  virtual llvm::StringRef getDescription() const { return ""; }
  Target *getParent() { return parent; }
  const Target *getParent() const { return parent; }

  template <class TargetType> std::vector<TargetType *> getChildrenOfType() const {
    std::vector<TargetType *> filteredChildren;
    for (auto &child : getChildren_()) {
      if (auto casted = dynamic_cast<TargetType*>(child.get()))
        filteredChildren.push_back(casted);
    }
    return filteredChildren;
  }

  std::vector<Target *> getChildren() const {
    return getChildrenOfType<Target>();
  }

  std::vector<TargetSystem *> getSubsystems() const {
    return getChildrenOfType<TargetSystem>();
  }

  std::vector<TargetInstrument *> getInstruments() const {
    return getChildrenOfType<TargetInstrument>();
  }

  template <class TargetType> size_t getNumTargetsOfType() const {
    return getChildrenOfType<TargetType>().size();
  }

  size_t getNumChildren() const {
    return getNumTargetsOfType<Target>();
  }

  size_t getNumInstruments() const {
    return getNumTargetsOfType<TargetInstrument>();
  }

  size_t getNumSubsystems() const {
    return getNumTargetsOfType<TargetSystem>();
  }

  virtual llvm::Error addPasses(mlir::PassManager &pm) = 0;
  virtual llvm::Error emitToPayload(mlir::ModuleOp &moduleOp, payload::Payload &payload) = 0;

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

  void addChild(std::unique_ptr<Target> child) {
    children_.push_back(std::move(child));
  }

  virtual llvm::Optional<qssc::arguments::BindArgumentsImplementationFactory *>
  getBindArgumentsImplementationFactory() {
    return llvm::None;
  };

  virtual ~TargetSystem() = default;

protected:
  virtual const std::vector<std::unique_ptr<Target>> &
  getChildren_() const override {
    return children_;
  }

  std::vector<std::unique_ptr<Target>> children_;
  // Helper containers to track instruments for quick access.
  std::vector<TargetSystem *> systems_;
  std::vector<TargetInstrument *> instruments_;
}; // class TargetSystem

class TargetInstrument : public Target {
protected: // Can only create subclasses
  TargetInstrument(std::string name, Target *parent);

  virtual const std::vector<std::unique_ptr<Target>> &
  getChildren_() const override {
    return children_;
  }

public:
  virtual ~TargetInstrument() = default;

private:
  // Used for returning empty children
  // TODO: Do something more graceful than this.
  static const std::vector<std::unique_ptr<Target>> children_;


}; // class TargetInstrument
} // namespace qssc::hal
#endif // TARGETSYSTEM_H
