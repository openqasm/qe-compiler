//===- MockTarget.cpp ----------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#include "MockTarget.h"

#include "Conversion/QUIRToLLVM/QUIRToLLVM.h"
#include "Conversion/QUIRToStandard/QUIRToStandard.h"
#include "Transforms/FunctionLocalization.h"
#include "Transforms/QubitLocalization.h"

#include "Dialect/QUIR/Transforms/Passes.h"
#include "HAL/TargetRegistry.h"
#include "HAL/TargetSystemRegistry.h"
#include "Payload/Payload.h"

#include "llvm/ADT/APFloat.h"
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
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>

using namespace mlir;
using namespace mlir::quir;

using namespace qssc::hal;
using namespace qssc::targets::mock;

// The space below at the front of the string causes this category to be printed
// first
static llvm::cl::OptionCategory
    mockCat(" QSS Compiler Options for the Mock target",
            "Options that control Mock-specific behavior of the Mock QSS "
            "Compiler target");

int qssc::targets::mock::init() {
    registry::TargetSystemRegistry::registerPlugin<MockSystem>(
            "mock", "Mock system for testing the targetting infrastructure.",
            [](llvm::Optional<llvm::StringRef> configurationPath)
                -> llvm::Expected<std::unique_ptr<hal::TargetSystem>> {
                if (!configurationPath)
                    return llvm::createStringError(
                        llvm::inconvertibleErrorCode(),
                        "Configuration file must be specified.\n");

            auto config = std::make_unique<MockConfig>(*configurationPath);
            return std::make_unique<MockSystem>(std::move(config));
       });

  registry::TargetRegistration<MockSystem> registrar(
      "mock", "Mock system for testing the targetting infrastructure.",
      [](llvm::Optional<llvm::StringRef> configurationPath)
          -> llvm::Expected<std::unique_ptr<hal::TargetSystem>> {
        if (!configurationPath)
          return llvm::createStringError(
              llvm::inconvertibleErrorCode(),
              "Configuration file must be specified.\n");

        auto config = std::make_unique<MockConfig>(*configurationPath);
        return std::make_unique<MockSystem>(std::move(config));
      });
  return 0;
}

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
    : TargetSystem("MockSystem", nullptr), mockConfig(std::move(config)) {
  instruments.push_back(
      std::make_unique<MockController>("MockController", this, *mockConfig));
  for (uint qubitId = 0; qubitId < mockConfig->getNumQubits(); ++qubitId) {
    instruments.push_back(std::make_unique<MockDrive>(
        "MockDrive_" + std::to_string(qubitId), this, *mockConfig));
  }
  for (uint acquireId = 0;
       acquireId <
       mockConfig->getNumQubits() / mockConfig->getMultiplexingRatio() + 1;
       ++acquireId) {
    instruments.push_back(std::make_unique<MockAcquire>(
        "MockAcquire_" + std::to_string(acquireId), this, *mockConfig));
  }
} // MockSystem

llvm::Error MockSystem::registerTargetPasses() {
  mlir::PassRegistration<qssc::targets::mock::MockFunctionLocalizationPass>();
  mlir::PassRegistration<qssc::targets::mock::MockQubitLocalizationPass>();
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
  pm.addPass(std::make_unique<SymbolTableBuildPass>());
  OpPassManager &nestedModulePM = pm.nest<ModuleOp>();
  nestedModulePM.addPass(std::make_unique<MockFunctionLocalizationPass>());
  nestedModulePM.addPass(
      std::make_unique<mlir::quir::FunctionArgumentSpecializationPass>());
} // mockPipelineBuilder
} // anonymous namespace

llvm::Error MockSystem::registerTargetPipelines() {
  mlir::PassPipelineRegistration<> pipeline(
      "mock-conversion", "Run Mock-specific conversions", mockPipelineBuilder);
  MockController::registerTargetPipelines();
  MockAcquire::registerTargetPipelines();
  MockDrive::registerTargetPipelines();

  return llvm::Error::success();
} // MockSystem::registerTargetPipelines

llvm::Error MockSystem::addPayloadPasses(mlir::PassManager &pm) {
  if (payloadPassesFound(pm)) {
    // command line specified payload conversion,
    // let the user handle exactly what to add
    return llvm::Error::success();
  }
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<BreakResetPass>());
  mockPipelineBuilder(pm);
  for (auto &child : children)
    if (auto err = child->addPayloadPasses(pm))
      return err;
  for (auto &instrument : instruments)
    if (auto err = instrument->addPayloadPasses(pm))
      return err;
  return llvm::Error::success();
} // MockSystem::addPayloadPasses

auto MockSystem::payloadPassesFound(mlir::PassManager &pm) -> bool {
  for (auto &pass : pm.getPasses())
    if (pass.getName() == "qssc::targets::mock::conversion::QUIRToStdPass")
      return true;
  return false;
} // MockSystem::payloadPassesFound

llvm::Error MockSystem::addToPayload(mlir::ModuleOp &moduleOp,
                                     qssc::payload::Payload &payload) {
  for (auto &child : children)
    if (auto err = child->addToPayload(moduleOp, payload))
      return err;
  for (auto &instrument : instruments)
    if (auto err = instrument->addToPayload(moduleOp, payload))
      return err;
  return llvm::Error::success();
} // MockSystem::addToPayload

MockController::MockController(std::string name, MockSystem *parent,
                               const SystemConfiguration &config)
    : TargetInstrument(std::move(name), parent) {} // MockController

void MockController::registerTargetPasses() {
} // MockController::registerTargetPasses

void MockController::registerTargetPipelines() {
} // MockController::registerTargetPipelines

llvm::Error MockController::addPayloadPasses(mlir::PassManager &pm) {
  return llvm::Error::success();
} // MockController::addPayloadPasses

auto MockController::getModule(ModuleOp topModuleOp) -> ModuleOp {
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
} // MockController::getModule

llvm::Error MockController::addToPayload(mlir::ModuleOp &moduleOp,
                                         qssc::payload::Payload &payload) {
  ModuleOp controllerModule = getModule(moduleOp);
  if (!controllerModule)
    controllerModule = moduleOp;
  auto *mlirStr = payload.getFile(name + ".mlir");
  llvm::raw_string_ostream mlirOStream(*mlirStr);
  mlirOStream << controllerModule;

  buildLLVMPayload(controllerModule, payload);

  return llvm::Error::success();
} // MockController::addToPayload

void MockController::buildLLVMPayload(mlir::ModuleOp &controllerModule,
                                      qssc::payload::Payload &payload) {

  auto *context = controllerModule.getContext();
  assert(context);

  // Register LLVM dialect and all infrastructure required for translation to
  // LLVM IR
  mlir::registerLLVMDialectTranslation(*context);

  mlir::PassManager pm(context);
  pm.addPass(std::make_unique<conversion::MockQUIRToStdPass>(false));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createLowerToLLVMPass());
  pm.addPass(mlir::LLVM::createLegalizeForExportPass());
  if (failed(pm.run(controllerModule))) {
    llvm::errs()
        << "Problems converting `MockController` module to std dialect!\n";
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

} // MockController::buildLLVMPayload

MockAcquire::MockAcquire(std::string name, MockSystem *parent,
                         const SystemConfiguration &config)
    : TargetInstrument(std::move(name), parent) {} // MockAcquire

auto MockAcquire::getModule(ModuleOp topModuleOp) -> ModuleOp {
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
} // MockAcquire::getModule

void MockAcquire::registerTargetPasses() {} // MockAcquire::registerTargetPasses

void MockAcquire::registerTargetPipelines() {
} // MockAcquire::registerTargetPipelines

llvm::Error MockAcquire::addPayloadPasses(mlir::PassManager &pm) {
  return llvm::Error::success();
} // MockAcquire::addPayloadPasses

llvm::Error MockAcquire::addToPayload(mlir::ModuleOp &moduleOp,
                                      qssc::payload::Payload &payload) {
  ModuleOp mockModule = getModule(moduleOp);
  if (!mockModule)
    mockModule = moduleOp;
  auto *mlirStr = payload.getFile(name + ".mlir");
  llvm::raw_string_ostream mlirOStream(*mlirStr);
  mlirOStream << mockModule;
  return llvm::Error::success();
} // MockAcquire::addToPayload

MockDrive::MockDrive(std::string name, MockSystem *parent,
                     const SystemConfiguration &config)
    : TargetInstrument(std::move(name), parent) {} // MockDrive

auto MockDrive::getModule(ModuleOp topModuleOp) -> ModuleOp {
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

void MockDrive::registerTargetPasses() {} // MockDrive::registerTargetPasses

void MockDrive::registerTargetPipelines() {
} // MockDrive::registerTargetPipelines

llvm::Error MockDrive::addPayloadPasses(mlir::PassManager &pm) {
  return llvm::Error::success();
} // MockDrive::addPayloadPasses

llvm::Error MockDrive::addToPayload(mlir::ModuleOp &moduleOp,
                                    qssc::payload::Payload &payload) {
  ModuleOp mockModule = getModule(moduleOp);
  if (!mockModule)
    mockModule = moduleOp;
  auto *mlirStr = payload.getFile(name + ".mlir");
  llvm::raw_string_ostream mlirOStream(*mlirStr);
  mlirOStream << mockModule;
  return llvm::Error::success();
} // MockDrive::addToPayload
