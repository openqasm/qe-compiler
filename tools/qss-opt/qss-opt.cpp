//===- qss-opt.cpp ----------------------------------------------*- C++ -*-===//
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
//
// This file implements an mlir-opt clone with the qss dialects and passes
// registered. It is mainly for diagnostic verification (testing) but can
// also be used for benchmarking and other types of testing.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "QSSC.h"

#include "HAL/PassRegistration.h"
#include "HAL/TargetRegistry.h"

#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/Pulse/Transforms/Passes.h"
#include "Dialect/QCS/IR/QCSDialect.h"
#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/Transforms/Passes.h"

using namespace qssc::hal;

static llvm::cl::opt<std::string> configurationPath(
    "config",
    llvm::cl::desc("Path to configuration file or directory (depends on the "
                   "target), - means use the config service"),
    llvm::cl::value_desc("path"));

static llvm::cl::opt<std::string>
    targetStr("target",
              llvm::cl::desc(
                  "Target architecture. Required for machine code generation."),
              llvm::cl::value_desc("targetName"));

static llvm::cl::opt<bool>
    addTargetPasses("add-target-passes",
                    llvm::cl::desc("Add target-specific passes"),
                    llvm::cl::init(false));

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently, delimited by '// -----'"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> verifyPasses(
    "verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true));

static llvm::cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    llvm::cl::desc("Allow operation with no registered dialects"),
    llvm::cl::init(false));

auto main(int argc, char **argv) -> int {
  mlir::registerAllPasses();
  mlir::registerConversionPasses();
  mlir::quir::registerQuirPasses();
  mlir::quir::registerQuirPassPipeline();
  mlir::pulse::registerPulsePasses();
  mlir::pulse::registerPulsePassPipeline();

  if (auto err = registerTargetPasses()) {
    llvm::errs() << err;
    return 1;
  }

  if (auto err = registerTargetPipelines()) {
    llvm::errs() << err;
    return 1;
  }

  mlir::DialectRegistry registry;
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  registerAllDialects(registry);
  registry.insert<mlir::quir::QUIRDialect, mlir::pulse::PulseDialect,
                  mlir::qcs::QCSDialect>();

  llvm::InitLLVM y(argc, argv);

  // Register any command line options.
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");

  // Build the list of dialects as a header for the --help message.
  std::string helpHeader = "qss-opt\n";
  {
    llvm::raw_string_ostream os(helpHeader);
    os << "Available Dialects: ";

    interleaveComma(registry.getDialectNames(), os,
                    [&](auto name) { os << name; });
    os << "\nAvailable Targets:\n";
    for (const auto &target : registry::registeredTargets()) {
      os << target.second.getTargetName() << " - "
         << target.second.getTargetDescription() << "\n";
    }
  }

  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::ParseCommandLineOptions(argc, argv, helpHeader);

  // Create target if one was specified.
  if (!targetStr.empty()) {
    auto targetInfo = registry::lookupTargetInfo(targetStr);
    if (!targetInfo) {
      llvm::errs() << "Error: Target " + targetStr + " is not registered.\n";
      return 1;
    }

    llvm::Optional<llvm::StringRef> conf{};
    if (!configurationPath.empty())
      conf.emplace(configurationPath);

    // Passing nullptr for context here registers the created target
    // as the default value for when an unknown MLIRContext* is passed to
    // TargetInfo::getTarget.
    // We do this only because MlirOptMain does not expose the MLIRContext
    // it creates for us.
    if (!targetInfo.getValue()->createTarget(nullptr, conf)) {
      llvm::errs() << "Error: Target " + targetStr + " could not be created.\n";
      return 1;
    }
  }

  // Set up the input file.
  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  if (failed(mlir::MlirOptMain(output->os(), std::move(file), passPipeline,
                               registry, splitInputFile, verifyDiagnostics,
                               verifyPasses, allowUnregisteredDialects, false)))
    return 1;

  // Keep the output file if the invocation of MlirOptMain was successful.
  output->keep();
  return 0;
}
