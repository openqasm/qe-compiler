//===- bind.cpp -----------------------------------------*- C++ -*-===//
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
#include <fstream>
#include <iostream>

#include <nlohmann/json.hpp>

#include "API/api.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Arguments/Arguments.h"

#include "HAL/PassRegistration.h"
#include "HAL/TargetSystem.h"
#include "HAL/TargetSystemRegistry.h"

using namespace mlir;

using json = nlohmann::json;

static llvm::cl::OptionCategory qsscBindCat_(" Options for parameter binding",
                                             "Options that binds parameters");

namespace {
enum GenAction { None, GenQEM, GenQEQEM };
} // anonymous namespace

static llvm::cl::opt<enum GenAction> emitGenAction(
    "emit", llvm::cl::init(GenAction::None),
    llvm::cl::desc("Select method used to generate input"),
    // llvm::cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    // llvm::cl::values(clEnumValN(DumpASTPretty, "ast-pretty",
    //                             "pretty print the AST")),
    // llvm::cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    // llvm::cl::values(clEnumValN(DumpWaveMem, "wavemem",
    //                             "output the waveform memory")),
    llvm::cl::values(clEnumValN(GenAction::GenQEM, "qem",
                                "a quantum executable module (qem) "
                                "for execution on hardware")),
    llvm::cl::values(
        clEnumValN(GenAction::GenQEQEM, "qe-qem",
                   "a target-specific quantum executable module (qeqem) "
                   "for execution on hardware")));

static llvm::cl::opt<std::string>
    inputQemFile(llvm::cl::Positional, llvm::cl::desc("[Input qem filename]"),
                 llvm::cl::init("-"), llvm::cl::cat(qsscBindCat_));

static llvm::cl::opt<std::string>
    outputQemFile("o", llvm::cl::desc("Output qem filename"),
                  llvm::cl::value_desc("filename"), llvm::cl::init("-"),
                  llvm::cl::cat(qsscBindCat_));

static llvm::cl::opt<std::string>
    paramFile("p", llvm::cl::desc("parameter json filename"),
              llvm::cl::value_desc("filename"), llvm::cl::init("-"),
              llvm::cl::cat(qsscBindCat_));

static llvm::cl::opt<std::string>
    targetStr("target",
              llvm::cl::desc("Target architecture used to generate inputfile."),
              llvm::cl::value_desc("targetName"), llvm::cl::init("-"),
              llvm::cl::cat(qsscBindCat_));

static llvm::cl::opt<std::string>
    configPathStr("config",
                  llvm::cl::desc("configuration path for target."),
                  llvm::cl::value_desc("configPath"), llvm::cl::init("-"),
                  llvm::cl::cat(qsscBindCat_));

llvm::Error loadJsonFile(json &d, const std::string &filename) {
  std::ifstream f(filename);
  if (!f.is_open()) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Unable to load json file: " + filename);
  } else {
    f >> d;
    return llvm::Error::success();
  }
}

class MapAngleArgumentSource : public qssc::arguments::ArgumentSource {

public:
  MapAngleArgumentSource(
      const std::unordered_map<std::string, double> &parameterMap)
      : parameterMap(parameterMap) {}

  qssc::arguments::ArgumentType
  getArgumentValue(llvm::StringRef name) const override {
    std::string name_{name};
    auto pos = parameterMap.find(name_);

    if (pos == parameterMap.end())
      return llvm::None;
    return pos->second;
  }

private:
  const std::unordered_map<std::string, double> &parameterMap;
};

llvm::Error
_bindArguments(std::string_view target, std::string_view configPath,
               std::string_view moduleInput, std::string_view payloadOutputPath,
               std::unordered_map<std::string, double> const &arguments,
               bool treatWarningsAsErrors, bool enableInMemoryInput,
               std::string *inMemoryOutput,
               const std::optional<qssc::DiagnosticCallback> &onDiagnostic) {

  MLIRContext context{};

  qssc::hal::registry::TargetSystemInfo &targetInfo =
      *qssc::hal::registry::TargetSystemRegistry::lookupPluginInfo(target)
           .getValueOr(qssc::hal::registry::TargetSystemRegistry::
                           nullTargetSystemInfo());

  auto created = targetInfo.createTarget(&context, llvm::StringRef(configPath));
  if (auto err = created.takeError()) {
    return llvm::joinErrors(
        llvm::createStringError(llvm::inconvertibleErrorCode(),
                                "Unable to create target!"),
        std::move(err));
  }

  auto targetInst = targetInfo.getTarget(&context);
  if (auto err = targetInst.takeError()) {
    return llvm::joinErrors(
        llvm::createStringError(llvm::inconvertibleErrorCode(),
                                "Unable to load target!"),
        std::move(err));
  }

  MapAngleArgumentSource source(arguments);

  auto factory = targetInst.get()->getBindArgumentsImplementationFactory();
  if ((!factory.hasValue()) || (factory.getValue() == nullptr)) {
    return qssc::emitDiagnostic(
        onDiagnostic, qssc::Severity::Error,
        qssc::ErrorCategory::QSSLinkerNotImplemented,
        "Unable to load bind arguments implementation for target.");
  }
  qssc::arguments::BindArgumentsImplementationFactory &factoryRef =
      *factory.getValue();
  return qssc::arguments::bindArguments(
      moduleInput, payloadOutputPath, source, treatWarningsAsErrors,
      enableInMemoryInput, inMemoryOutput, factoryRef, onDiagnostic);
}

int qssc::bindArguments(
    std::string_view target, std::string_view configPath,
    std::string_view moduleInput, std::string_view payloadOutputPath,
    std::unordered_map<std::string, double> const &arguments,
    bool treatWarningsAsErrors, bool enableInMemoryInput,
    std::string *inMemoryOutput,
    const std::optional<qssc::DiagnosticCallback> &onDiagnostic) {

  if (auto err =
          _bindArguments(target, configPath, moduleInput, payloadOutputPath,
                         arguments, treatWarningsAsErrors, enableInMemoryInput,
                         inMemoryOutput, onDiagnostic)) {
    llvm::logAllUnhandledErrors(std::move(err), llvm::errs());
    return 1;
  }
  return 0;
}

llvm::Error bind_(int argc, char const **argv, std::string *outputString,
                  std::optional<qssc::DiagnosticCallback> diagnosticCb) {
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Quantum System Software (QSS) Parameter Binding\n");

  if (paramFile == "-") {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Unable to detect a parameter file! Please specify the "
        "a parameter file with -p");
  }

  json inputArgs;
  if (loadJsonFile(inputArgs, paramFile)) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Unable to load a json file for arguments: file=" + paramFile);
  }

  auto argsMap = std::unordered_map<std::string, double>();
  for (auto &[paramName, paramValue] : inputArgs.items())
    argsMap[paramName] = (double)paramValue;

  if (emitGenAction == GenAction::None) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Unable to detect an emit option! Please specify the "
        "an emmit option with --emit=qem or --emit=qe-qem");
  }

  if (outputQemFile == "-") {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Unable to detect an output file! Please specify the "
        "a parameter file with -o");
  }

  if (inputQemFile == "-") {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Unable to detect an input file! Please specify the input qem file");
  }

  if (targetStr == "-") {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Unable to detect target used to generate "
                                   "input file! Please specify the "
                                   "target name with --target");
  }

  if (configPathStr == "-") {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Unable to detect configuration for the target! Please specify the "
        "configuration file path with --config");
  }

  bool treatWarningsAsErrors = false;
  bool enableInMemoryInput = false;

  return _bindArguments(targetStr, configPathStr, inputQemFile, outputQemFile,
                        argsMap, treatWarningsAsErrors, enableInMemoryInput,
                        nullptr, std::nullopt);
}

int qssc::bind(int argc, char const **argv, std::string *outputString,
               std::optional<qssc::DiagnosticCallback> diagnosticCb) {

  if (auto err = bind_(argc, argv, outputString, std::move(diagnosticCb))) {
    llvm::logAllUnhandledErrors(std::move(err), llvm::errs(), "Error: ");
    return 1;
  }

  return 0;
}
