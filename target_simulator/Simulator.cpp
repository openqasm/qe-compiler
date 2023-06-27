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
#include "Conversion/QUIRToStandard/QUIRToStandard.h"
#include "Transforms/FunctionLocalization.h"
#include "Transforms/QubitLocalization.h"

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

const std::vector<std::string> SimulatorSystem::childNames = {"SimulatorChild1",
                                                         "SimulatorChild2"};

SimulatorConfig::SimulatorConfig(llvm::StringRef configurationPath)
    : SystemConfiguration() {
  std::ifstream configStream(configurationPath.str());
  if (!configStream || !configStream.good()) {
    llvm::errs() << "Problem opening file " + configurationPath;
    return;
  }

  // This is a terrible parsing design just to make things work for now
  std::string fieldName;
  configStream >> fieldName;
  if (fieldName == "num_qubits") {
    configStream >> numQubits;
  } else {
    llvm::errs()
        << "Problem parsing configStream, expecting num_qubits and found "
        << fieldName << "\n";
  }
  configStream >> fieldName;
  if (fieldName == "acquire_multiplexing_ratio_to_1") {
    configStream >> multiplexing_ratio;
  } else {
    llvm::errs() << "Problem parsing configStream, expecting "
                    "acquire_multiplexing_ratio_to_1 and found "
                 << fieldName << "\n";
  }
  configStream >> fieldName;
  if (fieldName == "controllerNodeId") {
    configStream >> controllerNodeId;
  } else {
    llvm::errs()
        << "Problem parsing configStream, expecting controllerNodeId and found "
        << fieldName << "\n";
  }

  llvm::outs() << "Config:\nnum_qubits " << numQubits << "\nmultiplexing_ratio "
               << multiplexing_ratio << "\n";

  // preprocessing of config data for use by passes
  qubitDriveMap.resize(numQubits);
  qubitAcquireMap.resize(numQubits);
  uint nextId = 0, acquireId = 0;
  for (uint physId = 0; physId < numQubits; ++physId) {
    if (physId % multiplexing_ratio == 0) {
      acquireId = nextId++;
      qubitAcquireToPhysIdMap[acquireId] = std::vector<int>();
    }
    qubitAcquireToPhysIdMap[acquireId].push_back(physId);
    qubitAcquireMap[physId] = acquireId;
    qubitDriveMap[physId] = nextId++;
  }
} // SimulatorConfig

SimulatorSystem::SimulatorSystem(std::unique_ptr<SimulatorConfig> config)
    : TargetSystem("SimulatorSystem", nullptr), simulatorConfig(std::move(config)) {
  instruments.push_back(
      std::make_unique<SimulatorController>("SimulatorController", this, *simulatorConfig));
  for (uint qubitId = 0; qubitId < simulatorConfig->getNumQubits(); ++qubitId) {
    instruments.push_back(std::make_unique<SimulatorDrive>(
        "SimulatorDrive_" + std::to_string(qubitId), this, *simulatorConfig));
  }
  for (uint acquireId = 0;
       acquireId <
       simulatorConfig->getNumQubits() / simulatorConfig->getMultiplexingRatio() + 1;
       ++acquireId) {
    instruments.push_back(std::make_unique<SimulatorAcquire>(
        "SimulatorAcquire_" + std::to_string(acquireId), this, *simulatorConfig));
  }
} // SimulatorSystem

llvm::Error SimulatorSystem::registerTargetPasses() {
  mlir::PassRegistration<qssc::targets::simulator::SimulatorFunctionLocalizationPass>();
  mlir::PassRegistration<qssc::targets::simulator::SimulatorQubitLocalizationPass>();
  mlir::PassRegistration<conversion::SimulatorQUIRToStdPass>(
      []() -> std::unique_ptr<conversion::SimulatorQUIRToStdPass> {
        return std::make_unique<conversion::SimulatorQUIRToStdPass>(false);
      });
  SimulatorController::registerTargetPasses();
  SimulatorAcquire::registerTargetPasses();
  SimulatorDrive::registerTargetPasses();

  return llvm::Error::success();
} // SimulatorSystem::registerTargetPasses

namespace {
void simulatorPipelineBuilder(mlir::OpPassManager &pm) {
  pm.addPass(std::make_unique<mlir::quir::SubroutineCloningPass>());
  pm.addPass(std::make_unique<mlir::quir::RemoveQubitOperandsPass>());
  pm.addPass(std::make_unique<mlir::quir::ClassicalOnlyDetectionPass>());
  pm.addPass(std::make_unique<SimulatorQubitLocalizationPass>());
  pm.addPass(std::make_unique<SymbolTableBuildPass>());
  OpPassManager &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addPass(std::make_unique<SimulatorFunctionLocalizationPass>());
  nestedModulePM.addPass(
      std::make_unique<mlir::quir::FunctionArgumentSpecializationPass>());
} // simulatorPipelineBuilder
} // anonymous namespace

llvm::Error SimulatorSystem::registerTargetPipelines() {
  mlir::PassPipelineRegistration<> pipeline(
      "simulator-conversion", "Run Simulator-specific conversions", simulatorPipelineBuilder);
  SimulatorController::registerTargetPipelines();
  SimulatorAcquire::registerTargetPipelines();
  SimulatorDrive::registerTargetPipelines();

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
  return llvm::Error::success();
} // SimulatorSystem::addToPayload

SimulatorController::SimulatorController(std::string name, SimulatorSystem *parent,
                               const SystemConfiguration &config)
    : TargetInstrument(std::move(name), parent) {} // SimulatorController

void SimulatorController::registerTargetPasses() {
} // SimulatorController::registerTargetPasses

void SimulatorController::registerTargetPipelines() {
} // SimulatorController::registerTargetPipelines

llvm::Error SimulatorController::addPayloadPasses(mlir::PassManager &pm) {
  return llvm::Error::success();
} // SimulatorController::addPayloadPasses

auto SimulatorController::getModule(ModuleOp topModuleOp) -> ModuleOp {
  ModuleOp retOp = nullptr;
  topModuleOp->walk([&](ModuleOp walkOp) {
    auto nodeType = walkOp->getAttrOfType<StringAttr>("quir.nodeType");
    if (nodeType && nodeType.getValue() == "controller") {
      retOp = walkOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return retOp;
} // SimulatorController::getModule

llvm::Error SimulatorController::addToPayload(mlir::ModuleOp &moduleOp,
                                         qssc::payload::Payload &payload) {
  ModuleOp controllerModule = getModule(moduleOp);
  if (!controllerModule)
    controllerModule = moduleOp;
  auto *mlirStr = payload.getFile(name + ".mlir");
  llvm::raw_string_ostream mlirOStream(*mlirStr);
  mlirOStream << controllerModule;

  buildLLVMPayload(controllerModule, payload);

  return llvm::Error::success();
} // SimulatorController::addToPayload

void SimulatorController::buildLLVMPayload(mlir::ModuleOp &controllerModule,
                                      qssc::payload::Payload &payload) {

  auto *context = controllerModule.getContext();
  assert(context);

  // Register LLVM dialect and all infrastructure required for translation to
  // LLVM IR
  mlir::registerLLVMDialectTranslation(*context);

  mlir::PassManager pm(context);
  pm.addPass(std::make_unique<conversion::SimulatorQUIRToStdPass>(false));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createLowerToLLVMPass());
  pm.addPass(mlir::LLVM::createLegalizeForExportPass());
  if (failed(pm.run(controllerModule))) {
    llvm::errs()
        << "Problems converting `SimulatorController` module to std dialect!\n";
    return;
  }

  // Initialize native LLVM target
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeAllTargetMCs();

  // Setup the machine properties for the target architecture.
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

  if (auto err =
          quir::translateModuleToLLVMDialect(controllerModule, dataLayout)) {
    llvm::errs() << err;
    return;
  }

  // Build LLVM payload
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(controllerModule, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Error converting LLVM module to LLVM IR!\n";
    llvm::errs() << controllerModule << "\n";
    return;
  }

  llvmModule->setDataLayout(machine->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return;
  }
  std::string *payloadStr = payload.getFile("llvmModule.ll");
  llvm::raw_string_ostream llvmOStream(*payloadStr);
  llvmOStream << *llvmModule;

  // generate machine code and emit object file
  llvm::SmallString<128> objPath;
  int objFd;
  if (auto err = llvm::sys::fs::createTemporaryFile("controllerModule", "o",
                                                    objFd, objPath)) {
    llvm::errs()
        << "Failed to create temporary object file for controller module";
    return;
  }
  auto obj = std::make_unique<llvm::ToolOutputFile>(objPath, objFd);
  llvm::legacy::PassManager pass;

  if (machine->addPassesToEmitFile(pass, obj->os(), nullptr,
                                   llvm::CodeGenFileType::CGFT_ObjectFile)) {
    llvm::errs() << "Cannot emit object files with TargetMachine";
    return;
  }
  pass.run(*llvmModule);
  obj->os().flush();

  // Note: an actual target will likely invoke a linker and pull in libraries to
  // generate a binary, and possibly do more postprocessing steps to create a
  // binary that can be executed on the controller

  // include resulting file in payload
  std::ifstream binary{objPath.c_str(), std::ios_base::binary};

  if (!binary) {
    llvm::errs() << "Failed top open generated controller object file "
                 << objPath;
    return;
  }

  /* read whole content of object file into buffer */
  std::string binaryContents{
      std::istreambuf_iterator<char>(binary),
      /* eof representation */ std::istreambuf_iterator<char>()};

  payload.getFile("controller.bin")->assign(std::move(binaryContents));

} // SimulatorController::buildLLVMPayload

SimulatorAcquire::SimulatorAcquire(std::string name, SimulatorSystem *parent,
                         const SystemConfiguration &config)
    : TargetInstrument(std::move(name), parent) {} // SimulatorAcquire

auto SimulatorAcquire::getModule(ModuleOp topModuleOp) -> ModuleOp {
  ModuleOp retOp = nullptr;
  topModuleOp->walk([&](ModuleOp walkOp) {
    auto nodeType = walkOp->getAttrOfType<StringAttr>("quir.nodeType");
    if (nodeType && nodeType.getValue() == "acquire") {
      retOp = walkOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return retOp;
} // SimulatorAcquire::getModule

void SimulatorAcquire::registerTargetPasses() {} // SimulatorAcquire::registerTargetPasses

void SimulatorAcquire::registerTargetPipelines() {
} // SimulatorAcquire::registerTargetPipelines

llvm::Error SimulatorAcquire::addPayloadPasses(mlir::PassManager &pm) {
  return llvm::Error::success();
} // SimulatorAcquire::addPayloadPasses

llvm::Error SimulatorAcquire::addToPayload(mlir::ModuleOp &moduleOp,
                                      qssc::payload::Payload &payload) {
  ModuleOp simulatorModule = getModule(moduleOp);
  if (!simulatorModule)
    simulatorModule = moduleOp;
  auto *mlirStr = payload.getFile(name + ".mlir");
  llvm::raw_string_ostream mlirOStream(*mlirStr);
  mlirOStream << simulatorModule;
  return llvm::Error::success();
} // SimulatorAcquire::addToPayload

SimulatorDrive::SimulatorDrive(std::string name, SimulatorSystem *parent,
                     const SystemConfiguration &config)
    : TargetInstrument(std::move(name), parent) {} // SimulatorDrive

auto SimulatorDrive::getModule(ModuleOp topModuleOp) -> ModuleOp {
  ModuleOp retOp = nullptr;
  topModuleOp->walk([&](ModuleOp walkOp) {
    auto nodeType = walkOp->getAttrOfType<StringAttr>("quir.nodeType");
    if (nodeType && nodeType.getValue() == "drive") {
      retOp = walkOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return retOp;
}

void SimulatorDrive::registerTargetPasses() {} // SimulatorDrive::registerTargetPasses

void SimulatorDrive::registerTargetPipelines() {
} // SimulatorDrive::registerTargetPipelines

llvm::Error SimulatorDrive::addPayloadPasses(mlir::PassManager &pm) {
  return llvm::Error::success();
} // SimulatorDrive::addPayloadPasses

llvm::Error SimulatorDrive::addToPayload(mlir::ModuleOp &moduleOp,
                                    qssc::payload::Payload &payload) {
  ModuleOp simulatorModule = getModule(moduleOp);
  if (!simulatorModule)
    simulatorModule = moduleOp;
  auto *mlirStr = payload.getFile(name + ".mlir");
  llvm::raw_string_ostream mlirOStream(*mlirStr);
  mlirOStream << simulatorModule;
  return llvm::Error::success();
} // SimulatorDrive::addToPayload
