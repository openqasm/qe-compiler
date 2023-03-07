//===- SystemCreation.cpp ----------------------------------------*- C++-*-===//
//
// (C) Copyright IBM 2023.
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

#include "Dialect/Pulse/Transforms/SystemCreation.h"
#include "Utils/LegacyInputConversion.h"
#include "Utils/SystemDefinition.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include <filesystem>

namespace mlir::pulse {

using namespace qssc::utils;
namespace fs = ::std::filesystem;

void SystemCreationPass::runOnOperation() {

  markAllAnalysesPreserved();

  getAnalysis<LegacyInputConversion>();

  if (calibrationsFilename.hasValue() || expParamsFilename.hasValue() ||
      backendConfigFilename.hasValue()) {
    getCachedAnalysis<LegacyInputConversion>()->get().create(
        calibrationsFilename, expParamsFilename, backendConfigFilename);
  } else {
    llvm::errs() << "Encountered an error while checking input existence.";
  }
}

llvm::StringRef SystemCreationPass::getArgument() const {
  return "system-creation";
}
llvm::StringRef SystemCreationPass::getDescription() const {
  return "Create system graph model for the pulse domain.";
}

void SystemPlotPass::runOnOperation() {

  const auto system = getCachedAnalysis<LegacyInputConversion>();
  if (!system) {
    llvm::errs() << "Could not located cached values for System Definition.";
  } else {
    if (plotFilePath.hasValue()) {
      fs::path filepath = plotFilePath.getValue();
      std::ofstream output(filepath);
      if (!output) {
        llvm::errs() << "Encountered an error while trying to create a file at "
                     << filepath;
      }
      system->get().plot(output);
      output.close();
    } else {
      llvm::errs() << "Expected an filepath argument for SystemPlotPass.";
    }
  }
}

llvm::StringRef SystemPlotPass::getArgument() const { return "system-plot"; }
llvm::StringRef SystemPlotPass::getDescription() const {
  return "Plot system graph model for the pulse domain.";
}
} // namespace mlir::pulse
