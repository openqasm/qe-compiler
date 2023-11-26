//===- qss-opt.cpp ----------------------------------------------*- C++ -*-===//
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
// This file implements an mlir-opt clone with the qss dialects and passes
// registered. It is mainly for diagnostic verification (testing) but can
// also be used for benchmarking and other types of testing.
//
//===----------------------------------------------------------------------===//

#include "Dialect/OQ3/IR/OQ3Dialect.h"
#include "Dialect/OQ3/Transforms/Passes.h"
#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/Pulse/Transforms/Passes.h"
#include "Dialect/QCS/IR/QCSDialect.h"
#include "Dialect/QCS/Utils/ParameterInitialValueAnalysis.h"
#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/Transforms/Passes.h"
#include "Dialect/RegisterDialects.h"
#include "Dialect/RegisterPasses.h"
#include "HAL/PassRegistration.h"
#include "HAL/TargetSystemRegistry.h"
#include "Payload/PayloadRegistry.h"

#include "mlir/Debug/CLOptionsSetup.h"
#include "mlir/Debug/Counter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace qssc;
using namespace qssc::hal;

const static std::string toolName = "qss-opt";

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

llvm::Expected<std::pair<std::string, std::string>>
registerAndParseCLIOptions(int argc, char **argv,
                                 llvm::StringRef toolName,
                                 mlir::DialectRegistry &registry) {

  // Register any command line options.
  mlir::MlirOptMainConfig::registerCLOptions(registry);
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  mlir::tracing::DebugCounter::registerCLOptions();

  // Build the list of dialects as a header for the --help message.
  std::string helpHeader = (toolName + "\n").str();
  {
    llvm::raw_string_ostream os(helpHeader);
    os << "Available Dialects: ";

    interleaveComma(registry.getDialectNames(), os,
                    [&](auto name) { os << name; });
    os << "\nAvailable Targets:\n";
    for (const auto &target :
         registry::TargetSystemRegistry::registeredPlugins()) {
      os << target.second.getName() << " - " << target.second.getDescription()
         << "\n";
    }

    os << "\nAvailable Payloads:\n";
    for (const auto &payload :
         qssc::payload::registry::PayloadRegistry::registeredPlugins()) {
      os << payload.second.getName() << " - " << payload.second.getDescription()
         << "\n";
    }
  }

    // Create target if one was specified.
  if (!targetStr.empty()) {
    auto targetInfo =
        registry::TargetSystemRegistry::lookupPluginInfo(targetStr);
    if (!targetInfo)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Error: Target " + targetStr + " is not registered.\n");

    std::optional<llvm::StringRef> conf{};
    if (!configurationPath.empty())
      conf.emplace(configurationPath);

    // Passing nullptr for context here registers the created target
    // as the default value for when an unknown MLIRContext* is passed to
    // TargetInfo::getTarget.
    // We do this only because MlirOptMain does not expose the MLIRContext
    // it creates for us.
    if (!targetInfo.value()->createTarget(nullptr, conf))
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Error: Target " + targetStr + " could not be created.\n");
  }

  // Parse pass names in main to ensure static initialization completed.
  llvm::cl::ParseCommandLineOptions(argc, argv, helpHeader);
  return std::make_pair(inputFilename.getValue(), outputFilename.getValue());
}


mlir::LogicalResult QSSCOptMain(int argc, char **argv,
                                llvm::StringRef inputFilename,
                                llvm::StringRef outputFilename,
                                mlir::DialectRegistry &registry) {

  llvm::InitLLVM y(argc, argv);

  mlir::MlirOptMainConfig config = mlir::MlirOptMainConfig::createFromCLOptions();

  // When reading from stdin and the input is a tty, it is often a user mistake
  // and the process "appears to be stuck". Print a message to let the user know
  // about it!
  if (inputFilename == "-" &&
      llvm::sys::Process::FileDescriptorIsDisplayed(fileno(stdin)))
    llvm::errs() << "(processing input from stdin now, hit ctrl-c/ctrl-d to "
                    "interrupt)\n";

  // Set up the input file.
  std::string errorMessage;
  auto file = mlir::openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return mlir::failure();
  }

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return mlir::failure();
  }

  if (mlir::failed(mlir::MlirOptMain(output->os(), std::move(file), registry, config)))
    return mlir::failure();

  // Keep the output file if the invocation of MlirOptMain was successful.
  output->keep();
  return mlir::success();
}

auto main(int argc, char **argv) -> int {

  // Register the standard passes with MLIR.
  // Must precede the command line parsing.
  if (auto err = qssc::dialect::registerPasses()) {
    llvm::errs() << err << "\n";
    return 1;
  }

  mlir::DialectRegistry registry;
  qssc::dialect::registerDialects(registry);
  mlir::registerAllExtensions(registry);

  // Register and parse command line options.
  std::string inputFilename, outputFilename;
  auto expectedFileNames = registerAndParseCLIOptions(argc, argv, toolName, registry);
  if (auto err = expectedFileNames.takeError()) {
    llvm::errs() << err << "\n";
    return 1;
  }

  std::tie(inputFilename, outputFilename) = expectedFileNames.get();

  if (mlir::failed(QSSCOptMain(argc, argv, inputFilename, outputFilename, registry)))
    return 1;

  return 0;
}
