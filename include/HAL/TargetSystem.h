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

#include "API/api.h"
#include "API/errors.h"
#include "Arguments/Arguments.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/iterator_range.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include <list>
#include <optional>
#include <string>
#include <vector>

namespace qssc::payload {
class Payload;
} // namespace qssc::payload

namespace qssc::hal {
class Target;
class TargetSystem;
class TargetInstrument;

class Target {
protected:
  Target(std::string name, Target *parent);

  virtual const std::vector<std::unique_ptr<Target>> &getChildren_() const {
    return children_;
  };

public:
  virtual llvm::StringRef getName() const { return name; }
  virtual llvm::StringRef getDescription() const { return ""; }
  /// @brief Get the Target resource directory sub-path for this target
  /// which will be used for resolving external static resources at runtime
  /// that are configured through the build system.
  virtual llvm::StringRef getResourcePath() const { return getName(); }
  Target *getParent() { return parent; }
  const Target *getParent() const { return parent; }

  template <class TargetType>
  std::vector<TargetType *> getChildrenOfType() const {
    std::vector<TargetType *> filteredChildren;
    for (auto &child : getChildren_())
      if (auto casted = dynamic_cast<TargetType *>(child.get()))
        filteredChildren.push_back(casted);
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

  template <class TargetType>
  size_t getNumTargetsOfType() const {
    return getChildrenOfType<TargetType>().size();
  }

  size_t getNumChildren() const { return getNumTargetsOfType<Target>(); }

  size_t getNumInstruments() const {
    return getNumTargetsOfType<TargetInstrument>();
  }

  size_t getNumSubsystems() const {
    return getNumTargetsOfType<TargetSystem>();
  }

  /// @brief Lookup this target's module within the provided module operation.
  /// It is critical that the module be unique for the target be separate from
  /// any other module to ensure that it may be processed in parallel.
  /// @param parentModuleOp The parent module of this target. At this stage the
  /// parent must ensure that the child module has been created.
  /// @return The module operation for this target.
  virtual llvm::Expected<mlir::ModuleOp>
  getModule(mlir::ModuleOp parentModuleOp) = 0;
  /// @brief Configure the provided pass manager to process this target's
  /// module. It is important that all passes *only* touch this target's module
  /// to ensure MLIR's parallelization rules are obeyed.
  /// @param pm A pass manager that will operate *only* on this target's module
  virtual llvm::Error addPasses(mlir::PassManager &pm) = 0;
  /// @brief Compile and emit the target outputs to the supplied payload.
  /// This will also call and populate addPasses for this target and run the
  /// corresponding pass pipeline. Will be invoked *before* emitToPayload
  /// is called on any of its children.
  /// @param targetModuleOp The target module after application of the target's
  /// passes populated in addPasses and having run the pass manager on the
  /// module.
  /// @param payload The payload to populate for this target.
  virtual llvm::Error emitToPayload(mlir::ModuleOp targetModuleOp,
                                    payload::Payload &payload) = 0;
  /// @brief Hook called by the TargetCompilationManager
  /// after emitToPayload has been called on all children. This is useful
  /// for preparing amalgamated payload artifacts which require the children to
  /// have completed their payload emission. Will be invoked *after*
  /// emitToPayload has been called on all of its children.
  /// @param targetModuleOp The target module after application of the target's
  /// passes populated in addPasses and having run the pass manager on the
  /// module.
  /// @param payload The payload to populate for this target.
  virtual llvm::Error emitToPayloadPostChildren(mlir::ModuleOp targetModuleOp,
                                                payload::Payload &payload);

  virtual ~Target() = default;

  /// @brief Enable timing from this point for the target and its methods
  /// @param timingScope the root timer to nest timers from.
  void enableTiming(mlir::TimingScope &timingScope);
  /// @brief Disable(stop) ongoing timers
  void disableTiming();

  // Diagnostic creation and access
  /// @brief Add a diagnostic to this target
  void addDiagnostic(const qssc::Diagnostic &diag) {
    const std::lock_guard<std::mutex> lock(diagnosticsMutex_);
    diagnostics_.emplace_back(diag);
  }
  /// @brief Construct a diagnostic and add to this target
  void addDiagnostic(Severity severity, ErrorCategory category,
                     const std::string &message) {
    const std::lock_guard<std::mutex> lock(diagnosticsMutex_);
    diagnostics_.emplace_back(severity, category, message);
  }
  /// @brief Return the diagnostics from this target and its sub-targets.
  ///        Take and clear the diagnostic lists of the targets.
  qssc::DiagList takeDiagnostics() {
    const std::lock_guard<std::mutex> lock(diagnosticsMutex_);
    qssc::DiagList retDiagList;
    // Take the elements of the given list and move them to the calling list
    retDiagList.splice(retDiagList.end(), diagnostics_);

    for (auto &child : getChildren_())
      retDiagList.splice(retDiagList.end(), child->takeDiagnostics());

    return retDiagList;
  }

protected:
  /// @brief Get a nested timer instance from the root timer
  /// @param name The name of the timing span
  mlir::TimingScope getTimer(llvm::StringRef name);

  std::string name;

  // parent is already owned by unique_ptr
  // owned by the constructing context
  // class Target
  Target *parent;

  /// @brief Children targets storage.
  std::vector<std::unique_ptr<Target>> children_;

private:
  mlir::TimingScope rootTimer;

  /// @brief List of diagnostics generated for this target
  qssc::DiagList diagnostics_;

  /// @brief Mutex for adding diagnostics to the diagnostic list
  std::mutex diagnosticsMutex_;
};

class TargetSystem : public Target {
public:
  using PluginConfiguration = llvm::StringRef;

protected: // Can only create subclasses.
  TargetSystem(std::string name, Target *parent);

public:
  /// @brief Get the module for this system. Currently the default
  /// implementation simply returns the top-level module.
  /// TODO: In the future the method of system lookup should be more explicitly
  /// tied to the IR and the target.
  virtual llvm::Expected<mlir::ModuleOp>
  getModule(mlir::ModuleOp parentModuleOp) override;

  void addChild(std::unique_ptr<Target> child) {
    children_.push_back(std::move(child));
  }

  virtual std::optional<qssc::arguments::BindArgumentsImplementationFactory *>
  getBindArgumentsImplementationFactory(config::EmitAction action) {
    return std::nullopt;
  };

  llvm::Expected<TargetInstrument *> getInstrumentWithNodeId(uint nodeId) const;

  virtual ~TargetSystem() = default;

}; // class TargetSystem

class TargetInstrument : public Target {
protected: // Can only create subclasses
  TargetInstrument(std::string name, Target *parent);

public:
  virtual ~TargetInstrument() = default;

  /// @brief Get the module for this system. Currently the default
  /// implementation is based on lookup the module by node type and node id.
  /// TODO: In the future the method of system lookup should be more explicitly
  /// tied to the IR and the target.
  virtual llvm::Expected<mlir::ModuleOp>
  getModule(mlir::ModuleOp parentModuleOp) override;
  virtual llvm::StringRef getNodeType() = 0;
  virtual uint32_t getNodeId() = 0;

}; // class TargetInstrument
} // namespace qssc::hal
#endif // TARGETSYSTEM_H
