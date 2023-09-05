//===- Simulator.cpp ----------------------------------------*- C++ -*-===//
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

#include "AerSimulator.h"

#include "Conversion/QUIRToAer.h"
#include "Conversion/QUIRToLLVM/QUIRToLLVM.h"
#include "Transforms/OutputClassicalRegisters.h"

#include "Dialect/QUIR/Transforms/Passes.h"
#include "HAL/TargetSystemRegistry.h"
#include "Payload/Payload.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include <fstream>
#include <sstream>

using namespace mlir;
using namespace mlir::quir;

using namespace qssc::hal;
using namespace qssc::targets::simulator::aer;

// The space below at the front of the string causes this category to be printed
// first
static llvm::cl::OptionCategory simulatorCat(
    " QSS Compiler Options for the Simulator target",
    "Options that control Simulator-specific behavior of the Simulator QSS "
    "Compiler target");

int qssc::targets::simulator::aer::init() {
  bool registered =
      registry::TargetSystemRegistry::registerPlugin<AerSimulator>(
          "simulator",
          "Simulator system for testing the targeting infrastructure.",
          [](llvm::Optional<llvm::StringRef> configurationPath)
              -> llvm::Expected<std::unique_ptr<hal::TargetSystem>> {
            if (!configurationPath)
              return llvm::createStringError(
                  llvm::inconvertibleErrorCode(),
                  "Configuration file must be specified.\n");

            auto config =
                std::make_unique<AerSimulatorConfig>(*configurationPath);
            return std::make_unique<AerSimulator>(std::move(config));
          });
  return registered ? 0 : -1;
}

const std::vector<std::string> AerSimulator::childNames = {};

AerSimulatorConfig::AerSimulatorConfig(llvm::StringRef configurationPath)
    : SystemConfiguration() {} // SimulatorConfig

AerSimulator::AerSimulator(std::unique_ptr<AerSimulatorConfig> config)
    : TargetSystem("AerSimulator", nullptr),
      simulatorConfig(std::move(config)) {} // AerSimulator

llvm::Error AerSimulator::registerTargetPasses() {
  mlir::PassRegistration<transforms::OutputCRegsPass>(
      []() -> std::unique_ptr<transforms::OutputCRegsPass> {
        return std::make_unique<transforms::OutputCRegsPass>();
      });
  mlir::PassRegistration<conversion::QUIRToAERPass>(
      []() -> std::unique_ptr<conversion::QUIRToAERPass> {
        return std::make_unique<conversion::QUIRToAERPass>();
      });

  return llvm::Error::success();
} // AerSimulator::registerTargetPasses

namespace {
void simulatorPipelineBuilder(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<BreakResetPass>());
} // simulatorPipelineBuilder
} // anonymous namespace

llvm::Error AerSimulator::registerTargetPipelines() {
  mlir::PassPipelineRegistration<> pipeline(
      "simulator-conversion", "Run Simulator-specific conversions",
      simulatorPipelineBuilder);

  return llvm::Error::success();
} // AerSimulator::registerTargetPipelines

llvm::Error AerSimulator::addPayloadPasses(mlir::PassManager &pm) {
  if (payloadPassesFound(pm)) {
    // command line specified payload conversion,
    // let the user handle exactly what to add
    return llvm::Error::success();
  }

  simulatorPipelineBuilder(pm);

  return llvm::Error::success();
} // AerSimulator::addPayloadPasses

auto AerSimulator::payloadPassesFound(mlir::PassManager &pm) -> bool {
  for (auto &pass : pm.getPasses())
    if (pass.getName() == "qssc::targets::simulator::conversion::QUIRToStdPass")
      return true;
  return false;
} // AerSimulator::payloadPassesFound

llvm::Error AerSimulator::addToPayload(mlir::ModuleOp &moduleOp,
                                       qssc::payload::Payload &payload) {
  buildLLVMPayload(moduleOp, payload);

  // TODO: if buildLLVMPayload failed?
  return llvm::Error::success();
} // AerSimulator::addToPayload

void AerSimulator::buildLLVMPayload(mlir::ModuleOp &moduleOp,
                                    payload::Payload &payload) {

  auto *context = moduleOp.getContext();
  assert(context);

  // Register LLVM dialect and all infrastructure required for translation to
  // LLVM IR
  mlir::registerLLVMDialectTranslation(*context);

  mlir::PassManager pm(context);
  // Apply any generic pass manager command line options and run the pipeline.
  mlir::applyPassManagerCLOptions(pm);
  mlir::applyDefaultTimingPassManagerCLOptions(pm);

  // `OutputCRegsPass` must be applied before `VariableEliminationPass`.
  // It inserts classical `oq3` instructions for printing the values
  // of classical registers. These instructions will be converted into
  // standard ops by `VariableEliminationPass`.
  pm.addPass(std::make_unique<transforms::OutputCRegsPass>());
  pm.addPass(std::make_unique<quir::VariableEliminationPass>(false));
  pm.addPass(std::make_unique<conversion::QUIRToAERPass>());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createLowerToLLVMPass());
  pm.addPass(mlir::LLVM::createLegalizeForExportPass());
  if (failed(pm.run(moduleOp))) {
    llvm::errs() << "Problems converting `Simulator` module to AER!\n";
    return;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeAllTargetMCs();

  // Setup the machine properties for the target architecture.
  // TODO: In future, it would be better to make this configurable
  std::string targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string errorMessage;
  const auto *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!target) {
    llvm::errs() << "Unable to find target: " << errorMessage << "\n";
    return;
  }

  std::string cpu("generic");
  llvm::SubtargetFeatures features;
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetTriple, cpu, features.getString(), {}, {}));
  auto dataLayout = machine->createDataLayout();

  if (auto err = quir::translateModuleToLLVMDialect(moduleOp, dataLayout)) {
    llvm::errs() << err;
    return;
  }

  // Build LLVM payload
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(moduleOp, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Error converting LLVM module to LLVM IR!\n";
    llvm::errs() << moduleOp << "\n";
    return;
  }

  llvmModule->setDataLayout(dataLayout);
  llvmModule->setTargetTriple(targetTriple);

  // Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return;
  }

  llvm::SmallString<128> objPath;
  int objFd;
  if (auto err = llvm::sys::fs::createTemporaryFile("simulatorModule", "obj",
                                                    objFd, objPath)) {
    llvm::errs()
        << "Failed to create temporary object file for simulator module.\n";
    return;
  }
  auto obj = std::make_unique<llvm::ToolOutputFile>(objPath, objFd);
  llvm::legacy::PassManager pass;
  if (machine->addPassesToEmitFile(pass, obj->os(), nullptr,
                                   llvm::CodeGenFileType::CGFT_ObjectFile)) {
    llvm::errs() << "Cannot emit object files with TargetMachine.\n";
    return;
  }
  pass.run(*llvmModule);
  obj->os().flush();

  // Link the generated obj with a dynamic library of qiskit Aer
  char *LD = getenv("LD_PATH");
  char *AERLIB = getenv("LIBAER_PATH");

  llvm::SmallString<128> outputPath;
  if (auto EC = llvm::sys::fs::createTemporaryFile("simulatorModule", "out",
                                                   outputPath)) {
    return;
  }

  llvm::SmallVector<llvm::StringRef, 5> lld_argv{"ld", objPath, AERLIB, "-o",
                                                 outputPath};

  llvm::SmallString<128> stdErrPath;
  if (auto EC = llvm::sys::fs::createTemporaryFile("simulatorModule", "err",
                                                   stdErrPath)) {
    return;
  }

  llvm::Optional<llvm::StringRef> redirects[] = {
      {""}, {""}, llvm::StringRef(stdErrPath)};

  if (auto err = callTool(LD, lld_argv, redirects, true)) {
    auto bufOrError = llvm::MemoryBuffer::getFile(stdErrPath);
    if (!bufOrError) {
      llvm::errs() << "call linker error: " << bufOrError.getError().message()
                   << ", ret=" << err;
    } else {
      llvm::errs() << "call linker error: ret=" << err;
    }
    return;
  }

  std::ifstream output(outputPath.c_str(), std::ios_base::binary);
  if (!output) {
    llvm::errs() << "Failed to open generated simulator object file "
                 << outputPath;
    return;
  }

  std::string outputContents{std::istreambuf_iterator<char>(output),
                             std::istreambuf_iterator<char>()};

  payload.getFile("simulator.bin")->assign(std::move(outputContents));

} // AerSimulator::buildLLVMPayload

llvm::Error AerSimulator::callTool(
    llvm::StringRef program, llvm::ArrayRef<llvm::StringRef> args,
    llvm::ArrayRef<llvm::Optional<llvm::StringRef>> redirects, bool dumpArgs) {

  if (dumpArgs) {
    llvm::errs() << "Calling " << program << " with args";
    for (auto const &param : args)
      llvm::errs() << " " << param;
    llvm::errs() << "\n";
  }

  std::string executeError;
  int ret =
      llvm::sys::ExecuteAndWait(program, args, /* environment */ llvm::None,
                                redirects, 0, 0, &executeError);

  if (ret < 0 || executeError.size() > 0)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   llvm::Twine("Failed to execute ") + program +
                                       " " + executeError);

  if (ret == 0)
    return llvm::Error::success();
  else
    return llvm::createStringError(
        std::error_code{ret, std::generic_category()},
        "%*s failed with return code %d", program.size(), program.data(), ret);
} // callTool
