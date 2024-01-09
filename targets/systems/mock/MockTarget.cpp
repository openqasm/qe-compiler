//===- MockTarget.cpp ----------------------------------------*- C++ -*-===//
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

#include "MockTarget.h"

#include "API/errors.h"
#include "Conversion/QUIRToLLVM/QUIRToLLVM.h"
#include "Conversion/QUIRToStandard/QUIRToStandard.h"
#include "Dialect/QUIR/Transforms/BreakReset.h"
#include "Dialect/QUIR/Transforms/FunctionArgumentSpecialization.h"
#include "Dialect/QUIR/Transforms/Passes.h"
#include "Dialect/QUIR/Transforms/RemoveQubitOperands.h"
#include "Dialect/QUIR/Transforms/SubroutineCloning.h"
#include "HAL/SystemConfiguration.h"
#include "HAL/TargetSystem.h"
#include "HAL/TargetSystemRegistry.h"
#include "Payload/Payload.h"
#include "Transforms/QubitLocalization.h"

#include "mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"

#include <cstdint>
#include <fstream>
#include <ios>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <sys/types.h>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::quir;

using namespace qssc::hal;
using namespace qssc::targets::systems::mock;

namespace {
// The space below at the front of the string causes this category to be printed
// first
llvm::cl::OptionCategory
    mockCat(" QSS Compiler Options for the Mock target",
            "Options that control Mock-specific behavior of the Mock QSS "
            "Compiler target");
} // anonymous namespace

int qssc::targets::systems::mock::init() {
  bool const registered = registry::TargetSystemRegistry::registerPlugin<
      MockSystem>(
      "mock", "Mock system for testing the targeting infrastructure.",
      [](std::optional<std::pair<llvm::StringRef, qssc::OptDiagnosticCallback>>
             configurationPathAndCallback)
          -> llvm::Expected<std::unique_ptr<hal::TargetSystem>> {
        if (!configurationPathAndCallback.has_value())
          return llvm::createStringError(
              llvm::inconvertibleErrorCode(),
              "Configuration file must be specified.\n");

        auto configurationPath =
            std::get<0>(configurationPathAndCallback.value());
        auto config = std::make_unique<MockConfig>(configurationPath);
        return std::make_unique<MockSystem>(std::move(config));
      });
  return registered ? 0 : -1;
}

const std::vector<std::string> MockSystem::childNames = {"MockChild1",
                                                         "MockChild2"};

MockConfig::MockConfig(llvm::StringRef configurationPath)
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
} // MockConfig

MockSystem::MockSystem(std::unique_ptr<MockConfig> config)
    : TargetSystem("MockSystem", nullptr, std::nullopt),
      mockConfig(std::move(config)) {
  // Create controller target
  addChild(
      std::make_unique<MockController>("MockController", this, *mockConfig));

  // Create drive targets
  for (const auto &result : llvm::enumerate(mockConfig->getDriveNodes())) {
    uint32_t const qubitIdx = result.index();
    uint32_t const nodeId = result.value();
    addChild(std::make_unique<MockDrive>(
        "MockDrive_" + std::to_string(qubitIdx), this, *mockConfig, nodeId));
  }

  // Create acquire targets
  for (const auto &result : llvm::enumerate(mockConfig->getAcquireNodes())) {
    uint32_t const acquireIdx = result.index();
    uint32_t const nodeId = result.value();
    addChild(std::make_unique<MockAcquire>("MockAcquire_" +
                                               std::to_string(acquireIdx),
                                           this, *mockConfig, nodeId));
  }
} // MockSystem

llvm::Error MockSystem::registerTargetPasses() {
  mlir::PassRegistration<MockQubitLocalizationPass>();
  mlir::PassRegistration<conversion::MockQUIRToStdPass>(
      []() -> std::unique_ptr<conversion::MockQUIRToStdPass> {
        return std::make_unique<conversion::MockQUIRToStdPass>(false);
      });
  MockController::registerTargetPasses();
  MockAcquire::registerTargetPasses();
  MockDrive::registerTargetPasses();

  return llvm::Error::success();
} // MockSystem::registerTargetPasses

namespace {
void mockPipelineBuilder(mlir::OpPassManager &pm) {
  pm.addPass(std::make_unique<mlir::quir::SubroutineCloningPass>());
  pm.addPass(std::make_unique<mlir::quir::RemoveQubitOperandsPass>());
  pm.addPass(std::make_unique<mlir::quir::ClassicalOnlyDetectionPass>());
  pm.addPass(std::make_unique<MockQubitLocalizationPass>());
  OpPassManager &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addPass(
      std::make_unique<mlir::quir::FunctionArgumentSpecializationPass>());
} // mockPipelineBuilder
} // anonymous namespace

llvm::Error MockSystem::registerTargetPipelines() {
  mlir::PassPipelineRegistration<> const pipeline(
      "mock-conversion", "Run Mock-specific conversions", mockPipelineBuilder);
  MockController::registerTargetPipelines();
  MockAcquire::registerTargetPipelines();
  MockDrive::registerTargetPipelines();

  return llvm::Error::success();
} // MockSystem::registerTargetPipelines

llvm::Error MockSystem::addPasses(mlir::PassManager &pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<BreakResetPass>());
  mockPipelineBuilder(pm);

  return llvm::Error::success();
} // MockSystem::addPasses

llvm::Error MockSystem::emitToPayload(mlir::ModuleOp moduleOp,
                                      qssc::payload::Payload &payload) {
  return llvm::Error::success();
} // MockSystem::emitToPayload

MockController::MockController(std::string name, MockSystem *parent,
                               const SystemConfiguration &config)
    : TargetInstrument(std::move(name), parent) {} // MockController

void MockController::registerTargetPasses() {
} // MockController::registerTargetPasses

void MockController::registerTargetPipelines() {
} // MockController::registerTargetPipelines

llvm::Error MockController::addPasses(mlir::PassManager &pm) {
  pm.addPass(std::make_unique<conversion::MockQUIRToStdPass>(false));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::LLVM::createLegalizeForExportPass());

  return llvm::Error::success();
} // MockController::addPasses

llvm::Error MockController::emitToPayload(mlir::ModuleOp moduleOp,
                                          qssc::payload::Payload &payload) {

  auto *mlirStr = payload.getFile(name + ".mlir");
  llvm::raw_string_ostream mlirOStream(*mlirStr);
  mlirOStream << moduleOp;

  if (auto err = buildLLVMPayload(moduleOp, payload))
    return err;

  return llvm::Error::success();
} // MockController::emitToPayload

llvm::Error MockController::buildLLVMPayload(mlir::ModuleOp controllerModule,
                                             qssc::payload::Payload &payload) {
  auto timer = getTimer("build-llvm-payload");

  // Register LLVM dialect and all infrastructure required for translation to
  // LLVM IR
  auto initLLVMTimer = timer.nest("init-llvm");
  auto *context = controllerModule.getContext();
  mlir::registerBuiltinDialectTranslation(*context);
  mlir::registerLLVMDialectTranslation(*context);

  // Initialize native LLVM target
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmParser();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeAllTargetMCs();

  // Setup the machine properties for the target architecture.
  std::string const targetTriple = llvm::sys::getDefaultTargetTriple();
  std::string errorMessage;
  const auto *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, errorMessage);
  if (!target) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Unable to find target: " + errorMessage);
  }

  std::string const cpu("generic");
  llvm::SubtargetFeatures const features;
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetTriple, cpu, features.getString(), {}, {}));
  auto dataLayout = machine->createDataLayout();
  initLLVMTimer.stop();

  auto mlirToLLVMDialectTimer = timer.nest("translate-to-llvm-mlir-dialect");
  if (auto err =
          quir::translateModuleToLLVMDialect(controllerModule, dataLayout))
    return err;
  mlirToLLVMDialectTimer.stop();

  auto mlirToLLVMIRTimer = timer.nest("mlir-to-llvm-ir");
  // Build LLVM payload
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(controllerModule, llvmContext);
  if (!llvmModule) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Error converting LLVM module to LLVM IR!");
  }
  mlirToLLVMIRTimer.stop();

  auto llvmOptTimer = timer.nest("optimize-llvm");
  llvmModule->setDataLayout(machine->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);

  /// Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    return llvm::joinErrors(
        llvm::createStringError(llvm::inconvertibleErrorCode(),
                                "Failed to optimize LLVM IR"),
        std::move(err));
  }
  llvmOptTimer.stop();

  std::string *payloadStr = payload.getFile("llvmModule.ll");
  llvm::raw_string_ostream llvmOStream(*payloadStr);
  llvmOStream << *llvmModule;

  auto emitObjectFileTimer = timer.nest("build-object-file");
  // generate machine code and emit object file
  llvm::SmallString<128> objPath;
  int objFd;
  if (auto err = llvm::sys::fs::createTemporaryFile("controllerModule", "o",
                                                    objFd, objPath)) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Failed to create temporary object file for controller module");
  }
  auto obj = std::make_unique<llvm::ToolOutputFile>(objPath, objFd);
  llvm::legacy::PassManager pass;

  if (machine->addPassesToEmitFile(pass, obj->os(), nullptr,
                                   llvm::CodeGenFileType::CGFT_ObjectFile)) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Cannot emit object files with TargetMachine");
  }
  pass.run(*llvmModule);
  obj->os().flush();
  emitObjectFileTimer.stop();

  auto emitBinaryTimer = timer.nest("emit-binary");
  // Note: an actual target will likely invoke a linker and pull in libraries to
  // generate a binary, and possibly do more postprocessing steps to create a
  // binary that can be executed on the controller
  // include resulting file in payload
  std::ifstream binary{objPath.c_str(), std::ios_base::binary};

  if (!binary) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Failed top open generated controller object file" + objPath);
  }

  /* read whole content of object file into buffer */
  std::string binaryContents{
      std::istreambuf_iterator<char>(binary),
      /* eof representation */ std::istreambuf_iterator<char>()};

  payload.getFile("controller.bin")->assign(std::move(binaryContents));
  emitBinaryTimer.stop();

  return llvm::Error::success();

} // MockController::buildLLVMPayload

MockAcquire::MockAcquire(std::string name, MockSystem *parent,
                         const SystemConfiguration &config, uint32_t nodeId)
    : TargetInstrument(std::move(name), parent), nodeId_(nodeId) {
} // MockAcquire

void MockAcquire::registerTargetPasses() {} // MockAcquire::registerTargetPasses

void MockAcquire::registerTargetPipelines() {
} // MockAcquire::registerTargetPipelines

llvm::Error MockAcquire::addPasses(mlir::PassManager &pm) {
  return llvm::Error::success();
} // MockAcquire::addPasses

llvm::Error MockAcquire::emitToPayload(mlir::ModuleOp moduleOp,
                                       qssc::payload::Payload &payload) {
  std::string mlirStr;
  llvm::raw_string_ostream mlirOStream(mlirStr);
  mlirOStream << moduleOp;
  payload.getFile(name + ".mlir")->assign(mlirOStream.str());

  return llvm::Error::success();
} // MockAcquire::emitToPayload

MockDrive::MockDrive(std::string name, MockSystem *parent,
                     const SystemConfiguration &config, uint32_t nodeId)
    : TargetInstrument(std::move(name), parent), nodeId_(nodeId) {} // MockDrive

void MockDrive::registerTargetPasses() {} // MockDrive::registerTargetPasses

void MockDrive::registerTargetPipelines() {
} // MockDrive::registerTargetPipelines

llvm::Error MockDrive::addPasses(mlir::PassManager &pm) {
  return llvm::Error::success();
} // MockDrive::addPasses

llvm::Error MockDrive::emitToPayload(mlir::ModuleOp moduleOp,
                                     qssc::payload::Payload &payload) {

  std::string mlirStr;
  llvm::raw_string_ostream mlirOStream(mlirStr);
  mlirOStream << moduleOp;
  payload.getFile(name + ".mlir")->assign(mlirOStream.str());

  return llvm::Error::success();
} // MockDrive::emitToPayload
