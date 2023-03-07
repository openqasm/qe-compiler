//===- SystemCreation.h - Slice mlir using ports ------------------* C++-*-===//
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

#ifndef PULSE_LEGACY_SYSTEM_DEFINITION_H
#define PULSE_LEGACY_SYSTEM_DEFINITION_H

#include "Utils/SystemDefinition.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>

namespace mlir::pulse {

struct SystemCreationPass
    : public PassWrapper<SystemCreationPass, OperationPass<>> {

  SystemCreationPass() = default;

  SystemCreationPass(const SystemCreationPass &pass) : PassWrapper(pass) {}

  Option<std::string> inputType{
      *this, "input-type", llvm::cl::desc("Input type for setup creation"),
      llvm::cl::value_desc("input type"), llvm::cl::init("")};

  Option<std::string> calibrationsFilename{
      *this, "cal-file", llvm::cl::desc("Calibrations filename"),
      llvm::cl::value_desc("filename"), llvm::cl::init("")};
  Option<std::string> expParamsFilename{
      *this, "exp-params", llvm::cl::desc("Experiment Parameters filename"),
      llvm::cl::value_desc("filename"), llvm::cl::init("")};
  Option<std::string> backendConfigFilename{
      *this, "backend-config", llvm::cl::desc("Backend configuration"),
      llvm::cl::value_desc("filename"), llvm::cl::init("")};

  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
};

struct SystemPlotPass : public PassWrapper<SystemPlotPass, OperationPass<>> {

  SystemPlotPass() = default;

  SystemPlotPass(const SystemPlotPass &pass) : PassWrapper(pass) {}

  Option<std::string> plotFilePath{
      *this, "path", llvm::cl::desc("Plot fir path"),
      llvm::cl::value_desc("filename"), llvm::cl::init("")};

  void runOnOperation() override;

  llvm::StringRef getArgument() const override;
  llvm::StringRef getDescription() const override;
};

} // namespace mlir::pulse

#endif // PULSE_LEGACY_SYSTEM_DEFINITION_H
