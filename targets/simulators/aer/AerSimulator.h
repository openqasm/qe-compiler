//===- AerSimulator.h - Simulator target info -------------------*- C++ -*-===//
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
//  This file declares the classes for the a Simulator target interface
//
//===----------------------------------------------------------------------===//
#ifndef HAL_TARGETS_AER_SIMULATOR_H
#define HAL_TARGETS_AER_SIMULATOR_H

#include "HAL/SystemConfiguration.h"
#include "HAL/TargetSystem.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <unordered_map>

namespace qssc::targets::simulators::aer {

// Register the simulator target.
int init();

class AerSimulatorConfig : public qssc::hal::SystemConfiguration {
public:
  explicit AerSimulatorConfig(llvm::StringRef configurationPath);
}; // class AerSimulatorConfig

class AerSimulator : public qssc::hal::TargetSystem {
public:
  static constexpr auto name = "aer-simulator";
  static const std::vector<std::string> childNames;
  explicit AerSimulator(std::unique_ptr<AerSimulatorConfig> config);
  static llvm::Error registerTargetPasses();
  static llvm::Error registerTargetPipelines();
  llvm::Error addPayloadPasses(mlir::PassManager &pm) override;
  auto payloadPassesFound(mlir::PassManager &pm) -> bool;
  llvm::Error addToPayload(mlir::ModuleOp &moduleOp,
                           payload::Payload &payload) override;
  auto getConfig() -> AerSimulatorConfig & { return *simulatorConfig; }

  static llvm::Error
  callTool(llvm::StringRef program, llvm::ArrayRef<llvm::StringRef> args,
           llvm::ArrayRef<llvm::Optional<llvm::StringRef>> redirects,
           bool dumpArgs);

private:
  std::unique_ptr<AerSimulatorConfig> simulatorConfig;

private:
  void buildLLVMPayload(mlir::ModuleOp &moduleOp, payload::Payload &payload);
}; // class AerSimulatorSystem

} // namespace qssc::targets::simulators::aer

#endif // HAL_TARGETS_AER_SIMULATOR_H
