//===- AerSimulator.cpp -----------------------------------------*- C++ -*-===//
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

#include <nlohmann/json.hpp>

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
using namespace qssc::targets::simulators::aer;

using json = nlohmann::json;

namespace qssc::targets::simulators::aer {

int init() {
  bool registered =
      registry::TargetSystemRegistry::registerPlugin<AerSimulator>(
          "aer-simulator",
          "Quantum simulator using qiskit Aer for quantum programs in "
          "OpenQASM3/QUIR",
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

const char *toStringInAer(SimulationMethod method) {
  switch (method) {
  case SimulationMethod::STATEVECTOR:
    return "statevector";
  case SimulationMethod::DENSITY_MATRIX:
    return "density_matrix";
  case SimulationMethod::MPS:
    return "matrix_product_state";
  case SimulationMethod::STABILIZER:
    return "stabilizer";
  case SimulationMethod::EXTENDED_STABILIZER:
    return "extended_stabilizer";
  case SimulationMethod::UNITARY:
    return "unitary";
  case SimulationMethod::SUPEROP:
    return "superop";
  }

  assert(false && "Invalid simulation method");
  return "";
}

const char *toStringInAer(SimulationDevice device) {
  switch (device) {
  case SimulationDevice::CPU:
    return "CPU";
  case SimulationDevice::GPU:
    return "GPU";
  case SimulationDevice::THRUST_CPU:
    return "ThrustCPU";
  }

  assert(false && "Invalid simulation device");
  return "";
}

const char *toStringInAer(SimulationPrecision precision) {
  switch (precision) {
  case SimulationPrecision::DOUBLE:
    return "double";
  }

  assert(false && "Invalid simulation precision");
  return "";
}

} // namespace qssc::targets::simulators::aer

const std::vector<std::string> AerSimulator::childNames = {};

AerSimulatorConfig::AerSimulatorConfig(llvm::StringRef configurationPath)
    : SystemConfiguration(), method(SimulationMethod::STATEVECTOR),
      device(SimulationDevice::CPU), precision(SimulationPrecision::DOUBLE) {
  std::ifstream cfgFile(configurationPath.data());
  if (!cfgFile) {
    llvm::errs() << "Failed to open the configuration file: ";
    llvm::errs() << configurationPath;
    return;
  }

  json cfg;
  cfgFile >> cfg;
  if (cfg.contains("method")) {
    const auto cfgMethod = cfg["method"];
    if (cfgMethod == "statevector")
      method = SimulationMethod::STATEVECTOR;
    else if (cfgMethod == "density_matrix")
      method = SimulationMethod::DENSITY_MATRIX;
    else if (cfgMethod == "MPS")
      method = SimulationMethod::MPS;
    else if (cfgMethod == "stabilizer")
      method = SimulationMethod::STABILIZER;
    else if (cfgMethod == "extended_stabilizer")
      method = SimulationMethod::EXTENDED_STABILIZER;
    else if (cfgMethod == "unitary")
      method = SimulationMethod::UNITARY;
    else if (cfgMethod == "superop")
      method = SimulationMethod::SUPEROP;
    else {
      llvm::errs() << "Unsupported Aer simulation method: " << cfgMethod.dump();
      llvm::errs() << ". Use default value.\n";
    }
  }
  if (cfg.contains("device")) {
    const auto cfgDevice = cfg["device"];
    if (cfgDevice == "cpu" || cfgDevice == "CPU")
      device = SimulationDevice::CPU;
    else if (cfgDevice == "gpu" || cfgDevice == "GPU")
      device = SimulationDevice::GPU;
    else if (cfgDevice == "thrust_cpu")
      device = SimulationDevice::THRUST_CPU;
    else {
      llvm::errs() << "Unsupported Aer simulation device: " << cfgDevice.dump();
      llvm::errs() << ". Use default value.\n";
    }
  }
  if (cfg.contains("precision")) {
    const auto cfgPrecision = cfg["precision"];
    if (cfgPrecision == "double")
      precision = SimulationPrecision::DOUBLE;
    else {
      llvm::errs() << "Unsupported Aer simulation precision: "
                   << cfgPrecision.dump();
      llvm::errs() << ". Use default value.\n";
    }
  }
} // SimulatorConfig

AerSimulator::AerSimulator(std::unique_ptr<AerSimulatorConfig> config)
    : TargetSystem("AerSimulator", nullptr),
      simulatorConfig(std::move(config)) {} // AerSimulator

llvm::Error AerSimulator::registerTargetPasses() {
  mlir::PassRegistration<transforms::OutputCRegsPass>();
  mlir::PassRegistration<conversion::QUIRToAERPass>();

  return llvm::Error::success();
} // AerSimulator::registerTargetPasses

namespace {
void simulatorPipelineBuilder(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(std::make_unique<BreakResetPass>());
  // `OutputCRegsPass` must be applied before `VariableEliminationPass`.
  // It inserts classical `oq3` instructions for printing the values
  // of classical registers. These instructions will be converted into
  // standard ops by `VariableEliminationPass`.
  pm.addPass(std::make_unique<transforms::OutputCRegsPass>());
  pm.addPass(std::make_unique<quir::VariableEliminationPass>(false));
  pm.addPass(std::make_unique<conversion::QUIRToAERPass>());
} // simulatorPipelineBuilder
} // anonymous namespace

llvm::Error AerSimulator::registerTargetPipelines() {
  mlir::PassPipelineRegistration<> pipeline(
      "aer-simulator-conversion", "Run Aer simulator specific conversions",
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

bool AerSimulator::payloadPassesFound(mlir::PassManager &pm) {
  for (auto &pass : pm.getPasses())
    if (pass.getName() ==
        "qssc::targets::simulator::aer::conversion::QUIRToAerPass")
      return true;
  return false;
} // AerSimulator::payloadPassesFound

llvm::Error AerSimulator::addToPayload(mlir::ModuleOp &moduleOp,
                                       qssc::payload::Payload &payload) {
  return buildLLVMPayload(moduleOp, payload);
} // AerSimulator::addToPayload

// FUTURE: Support to lower completely to LLVM-IR and generate a binary file
llvm::Error AerSimulator::buildLLVMPayload(mlir::ModuleOp &moduleOp,
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

  pm.addPass(mlir::createLowerToLLVMPass());
  pm.addPass(mlir::LLVM::createLegalizeForExportPass());
  if (failed(pm.run(moduleOp))) {
    return llvm::make_error<llvm::StringError>(
        "Problems converting `Simulator` module to AER!\n",
        llvm::inconvertibleErrorCode());
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
    return llvm::make_error<llvm::StringError>(
        "Unable to find target: " + errorMessage + "\n",
        llvm::inconvertibleErrorCode());
  }

  std::string cpu("generic");
  llvm::SubtargetFeatures features;
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      targetTriple, cpu, features.getString(), {}, {}));
  auto dataLayout = machine->createDataLayout();

  if (auto err = quir::translateModuleToLLVMDialect(moduleOp, dataLayout))
    return err;

  // Build LLVM payload
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(moduleOp, llvmContext);
  if (!llvmModule) {
    std::string msg;
    llvm::raw_string_ostream os(msg);
    os << "Error converting LLVM module to LLVM IR!\n";
    os << moduleOp << "\n";
    return llvm::make_error<llvm::StringError>(msg,
                                               llvm::inconvertibleErrorCode());
  }

  llvmModule->setDataLayout(dataLayout);
  llvmModule->setTargetTriple(targetTriple);

  // Optionally run an optimization pipeline over the llvm module.
  auto optPipeline = mlir::makeOptimizingTransformer(0, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    std::string msg;
    llvm::raw_string_ostream os(msg);
    os << "Failed to optimize LLVM IR: " << err << "\n";
    return llvm::make_error<llvm::StringError>(msg,
                                               llvm::inconvertibleErrorCode());
  }

  llvm::SmallString<128> objPath;
  int objFd;
  if (auto err = llvm::sys::fs::createTemporaryFile("simulatorModule", "o",
                                                    objFd, objPath)) {
    return llvm::make_error<llvm::StringError>(
        "Failed to create temporary object file for simulator module.\n",
        llvm::inconvertibleErrorCode());
  }
  auto obj = std::make_unique<llvm::ToolOutputFile>(objPath, objFd);
  llvm::legacy::PassManager pass;
  if (machine->addPassesToEmitFile(pass, obj->os(), nullptr,
                                   llvm::CodeGenFileType::CGFT_ObjectFile)) {
    return llvm::make_error<llvm::StringError>(
        "Cannot emit object files with TargetMachine.\n",
        llvm::inconvertibleErrorCode());
  }
  pass.run(*llvmModule);
  obj->os().flush();

  // TODO: In future, we need to link the generated object file
  // with `libaer.{so, dylib}` to create a executable file here.
  // An external linker (e.g. ld) may have to be called and also
  // we have to specify the path to the linker and the shared library.

  std::ifstream binary(objPath.c_str(), std::ios_base::binary);
  if (!binary) {
    return llvm::make_error<llvm::StringError>(
        "Failed to open generated object file.",
        llvm::inconvertibleErrorCode());
  }

  std::string binaryContents{std::istreambuf_iterator<char>(binary),
                             std::istreambuf_iterator<char>()};

  payload.getFile("simulator.bin")->assign(std::move(binaryContents));

  return llvm::Error::success();
} // AerSimulator::buildLLVMPayload
