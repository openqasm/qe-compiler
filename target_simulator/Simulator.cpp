//===- SimulatorTarget.cpp ----------------------------------------*- C++ -*-===//
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

#include "Simulator.h"

#include "Conversion/QUIRToLLVM/QUIRToLLVM.h"
#include "Conversion/QUIRToAer.h"

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
using namespace qssc::targets::simulator;

// The space below at the front of the string causes this category to be printed
// first
static llvm::cl::OptionCategory
    simulatorCat(" QSS Compiler Options for the Simulator target",
            "Options that control Simulator-specific behavior of the Simulator QSS "
            "Compiler target");

int qssc::targets::simulator::init() {
  bool registered = registry::TargetSystemRegistry::registerPlugin<SimulatorSystem>(
      "simulator", "Simulator system for testing the targeting infrastructure.",
      [](llvm::Optional<llvm::StringRef> configurationPath)
          -> llvm::Expected<std::unique_ptr<hal::TargetSystem>> {
        if (!configurationPath)
          return llvm::createStringError(
              llvm::inconvertibleErrorCode(),
              "Configuration file must be specified.\n");

        auto config = std::make_unique<SimulatorConfig>(*configurationPath);
        return std::make_unique<SimulatorSystem>(std::move(config));
      });
  return registered ? 0 : -1;
}

const std::vector<std::string> SimulatorSystem::childNames =
    {"SimulatorChild1", "SimulatorChild2"};

SimulatorConfig::SimulatorConfig(llvm::StringRef configurationPath)
    : SystemConfiguration() {
} // SimulatorConfig

SimulatorSystem::SimulatorSystem(std::unique_ptr<SimulatorConfig> config)
    : TargetSystem("SimulatorSystem", nullptr), simulatorConfig(std::move(config)) {
} // SimulatorSystem

llvm::Error SimulatorSystem::registerTargetPasses() {
  mlir::PassRegistration<conversion::QUIRToAERPass>(
      []() -> std::unique_ptr<conversion::QUIRToAERPass> {
        return std::make_unique<conversion::QUIRToAERPass>(false);
      });

  return llvm::Error::success();
} // SimulatorSystem::registerTargetPasses

namespace {
void simulatorPipelineBuilder(mlir::OpPassManager &pm) {
} // simulatorPipelineBuilder
} // anonymous namespace

llvm::Error SimulatorSystem::registerTargetPipelines() {
  mlir::PassPipelineRegistration<> pipeline(
      "simulator-conversion", "Run Simulator-specific conversions", simulatorPipelineBuilder);

  return llvm::Error::success();
} // SimulatorSystem::registerTargetPipelines

llvm::Error SimulatorSystem::addPayloadPasses(mlir::PassManager &pm) {
  if (payloadPassesFound(pm)) {
    // command line specified payload conversion,
    // let the user handle exactly what to add
    return llvm::Error::success();
  }
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<BreakResetPass>());
  simulatorPipelineBuilder(pm);
  for (auto &child : children)
    if (auto err = child->addPayloadPasses(pm))
      return err;
  for (auto &instrument : instruments)
    if (auto err = instrument->addPayloadPasses(pm))
      return err;
  return llvm::Error::success();
} // SimulatorSystem::addPayloadPasses

auto SimulatorSystem::payloadPassesFound(mlir::PassManager &pm) -> bool {
  for (auto &pass : pm.getPasses())
    if (pass.getName() == "qssc::targets::simulator::conversion::QUIRToStdPass")
      return true;
  return false;
} // SimulatorSystem::payloadPassesFound

llvm::Error SimulatorSystem::addToPayload(mlir::ModuleOp &moduleOp,
                                     qssc::payload::Payload &payload) {
  for (auto &child : children)
    if (auto err = child->addToPayload(moduleOp, payload))
      return err;
  for (auto &instrument : instruments)
    if (auto err = instrument->addToPayload(moduleOp, payload))
      return err;

  buildLLVMPayload(moduleOp, payload);

  // TODO: if buildLLVMPayload failed?
  return llvm::Error::success();
} // SimulatorSystem::addToPayload

void SimulatorSystem::buildLLVMPayload(mlir::ModuleOp &moduleOp,
                                       payload::Payload &payload) {

  auto *context = moduleOp.getContext();
  assert(context);
  
  // Register LLVM dialect and all infrastructure required for translation to
  // LLVM IR
  mlir::registerLLVMDialectTranslation(*context);
  
  mlir::PassManager pm(context);
  pm.addPass(std::make_unique<conversion::QUIRToAERPass>(false));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createLowerToLLVMPass());
  pm.addPass(mlir::LLVM::createLegalizeForExportPass());
  if(failed(pm.run(moduleOp))) {
    llvm::errs() << "Problems converting `Simulator` module to AER!\n";
    return;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeAllTargetMCs();
  
  // Setup the machine properties for the target architecture.
  std::string targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string errorMessage;
  const auto *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if(!target) {
    llvm::errs() << "Unable to find target: " << errorMessage << "\n";
    return;
  }
  
  std::string cpu("generic");
  llvm::SubtargetFeatures features;
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetTriple, cpu, features.getString(), {}, {}));
  auto dataLayout = machine->createDataLayout();
  
  // TODO: Define a function like `aer::translateModuleToLLVMDialect`
  if(auto err =
        quir::translateModuleToLLVMDialect(moduleOp, dataLayout)) {
    llvm::errs() << err;
    return;
  }
  
  // Build LLVM payload
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(moduleOp, llvmContext);
  if(!llvmModule) {
    llvm::errs() << "Error converting LLVM module to LLVM IR!\n";
    llvm::errs() << moduleOp << "\n";
    return;
  }
  
  // TODO: why do not use created datalayout above?
  llvmModule->setDataLayout(machine->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);
  
  // Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return;
  }
  
  // TODO: debug output
  llvm::outs() << "output il file:\n";
  llvm::outs() << *llvmModule;
  
  llvm::SmallString<128> objPath;
  int objFd;
  if (auto err = llvm::sys::fs::createTemporaryFile("simulatorModule", "o", objFd, objPath)) {
    llvm::errs() << "Failed to create temporary object file for simulator module.\n";
    return;
  }
  auto obj = std::make_unique<llvm::ToolOutputFile>(objPath, objFd);
  llvm::legacy::PassManager pass;
  if(machine->addPassesToEmitFile(pass, obj->os(), nullptr,
                                  llvm::CodeGenFileType::CGFT_ObjectFile)) {
    llvm::errs() << "Cannot emit object files with TargetMachine.\n";
    return;
  }
  pass.run(*llvmModule);
  obj->os().flush();

  std::ifstream binary(objPath.c_str(), std::ios_base::binary);
  if(!binary) {
    llvm::errs() << "Failed to open generated simulator object file " << objPath;
    return;
  }
  
  std::string binaryContents{
    std::istreambuf_iterator<char>(binary),
    std::istreambuf_iterator<char>()};
  
  payload.getFile("simulator.bin")->assign(std::move(binaryContents));
} // SimulatorSystem::buildLLVMPayload