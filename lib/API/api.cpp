//===- api.cpp --------------------------------------------------*- C++ -*-===//
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

#include "API/api.h"
#include "Config/CLIConfig.h"
#include "Config/EnvVarConfig.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Arguments/Arguments.h"
#include "Payload/Payload.h"
#include "Payload/PayloadRegistry.h"
#include "QSSC.h"

#include "HAL/PassRegistration.h"
#include "HAL/TargetSystem.h"
#include "HAL/TargetSystemRegistry.h"

#include "Dialect/OQ3/Transforms/Passes.h"
#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/Pulse/Transforms/Passes.h"
#include "Dialect/QCS/Utils/ParameterInitialValueAnalysis.h"
#include "Dialect/QUIR/Transforms/Passes.h"
#include "Dialect/RegisterDialects.h"

#include "Frontend/OpenQASM3/OpenQASM3Frontend.h"

#include <filesystem>
#include <string_view>
#include <utility>

using namespace mlir;

static llvm::cl::opt<std::string> inputSource(
    llvm::cl::Positional, llvm::cl::desc("Input filename or program source"),
    llvm::cl::init("-"), llvm::cl::cat(qssc::config::getQSSCCategory()));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"),
                   llvm::cl::cat(qssc::config::getQSSCCategory()));

static llvm::cl::opt<bool>
    directInput("direct",
                llvm::cl::desc("Accept the input program directly as a string"),
                llvm::cl::cat(qssc::config::getQSSCCategory()));

static llvm::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false), llvm::cl::cat(qssc::config::getQSSCCategory()));

static llvm::cl::opt<bool> verifyPasses(
    "verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true), llvm::cl::cat(qssc::config::getQSSCCategory()));

static llvm::cl::opt<bool> showDialects(
    "show-dialects", llvm::cl::desc("Print the list of registered dialects"),
    llvm::cl::init(false), llvm::cl::cat(qssc::config::getQSSCCategory()));

static llvm::cl::opt<bool> showTargets(
    "show-targets", llvm::cl::desc("Print the list of registered targets"),
    llvm::cl::init(false), llvm::cl::cat(qssc::config::getQSSCCategory()));

static llvm::cl::opt<bool> showPayloads(
    "show-payloads", llvm::cl::desc("Print the list of registered payloads"),
    llvm::cl::init(false), llvm::cl::cat(qssc::config::getQSSCCategory()));

static llvm::cl::opt<bool> showConfig(
    "show-config", llvm::cl::desc("Print the loaded compiler configuration."),
    llvm::cl::init(false), llvm::cl::cat(qssc::config::getQSSCCategory()));

static llvm::cl::opt<bool> plaintextPayload(
    "plaintext-payload", llvm::cl::desc("Write the payload in plaintext"),
    llvm::cl::init(false), llvm::cl::cat(qssc::config::getQSSCCategory()));

static llvm::cl::opt<bool> includeSourceInPayload(
    "include-source", llvm::cl::desc("Write the input source into the payload"),
    llvm::cl::init(false), llvm::cl::cat(qssc::config::getQSSCCategory()));

namespace {
enum InputType { NONE, QASM, MLIR, QOBJ };
} // anonymous namespace
static llvm::cl::opt<enum InputType> inputType(
    "X", llvm::cl::init(InputType::NONE),
    llvm::cl::desc("Specify the kind of input desired"),
    llvm::cl::values(
        clEnumValN(InputType::QASM, "qasm",
                   "load the input file as an OpenQASM 3.0 source")),
    llvm::cl::values(clEnumValN(MLIR, "mlir",
                                "load the input file as an MLIR file")),
    llvm::cl::values(clEnumValN(QOBJ, "qobj",
                                "load the input file as a QOBJ file")));

namespace {
enum Action { None, DumpAST, DumpASTPretty, DumpMLIR, DumpWaveMem, GenQEM };
} // anonymous namespace
static llvm::cl::opt<enum Action> emitAction(
    "emit", llvm::cl::init(Action::None),
    llvm::cl::desc("Select the kind of output desired"),
    llvm::cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    llvm::cl::values(clEnumValN(DumpASTPretty, "ast-pretty",
                                "pretty print the AST")),
    llvm::cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    llvm::cl::values(clEnumValN(DumpWaveMem, "wavemem",
                                "output the waveform memory")),
    llvm::cl::values(clEnumValN(GenQEM, "qem",
                                "generate a quantum executable module (qem) "
                                "for execution on hardware")));

namespace qss {
enum FileExtension { None, AST, ASTPRETTY, QASM, QOBJ, MLIR, WMEM, QEM };
} // namespace qss

auto fileExtensionToStr(const qss::FileExtension &inExt) -> std::string {
  switch (inExt) {
  case qss::FileExtension::AST:
    return "ast";
    break;
  case qss::FileExtension::ASTPRETTY:
    return "ast-pretty";
    break;
  case qss::FileExtension::QASM:
    return "qasm";
    break;
  case qss::FileExtension::QOBJ:
    return "qobj";
    break;
  case qss::FileExtension::MLIR:
    return "mlir";
    break;
  case qss::FileExtension::WMEM:
    return "wmem";
    break;
  case qss::FileExtension::QEM:
    return "qem";
    break;
  default:
    return "none";
    break;
  }
  return "none";
}

auto fileExtensionToInputType(const qss::FileExtension &inExt) -> InputType {
  switch (inExt) {
  case qss::FileExtension::QASM:
    return InputType::QASM;
    break;
  case qss::FileExtension::QOBJ:
    return InputType::QOBJ;
    break;
  case qss::FileExtension::MLIR:
    return InputType::MLIR;
    break;
  default:
    break;
  }
  return InputType::NONE;
}

auto fileExtensionToAction(const qss::FileExtension &inExt) -> Action {
  switch (inExt) {
  case qss::FileExtension::AST:
    return Action::DumpAST;
    break;
  case qss::FileExtension::ASTPRETTY:
    return Action::DumpASTPretty;
    break;
  case qss::FileExtension::MLIR:
    return Action::DumpMLIR;
    break;
  case qss::FileExtension::WMEM:
    return Action::DumpWaveMem;
    break;
  case qss::FileExtension::QEM:
    return Action::GenQEM;
    break;
  default:
    break;
  }
  return Action::None;
}

auto strToFileExtension(const std::string &extStr) -> qss::FileExtension {
  if (extStr == "ast" || extStr == "AST")
    return qss::FileExtension::AST;
  if (extStr == "ast-pretty" || extStr == "AST-PRETTY")
    return qss::FileExtension::ASTPRETTY;
  if (extStr == "qasm" || extStr == "QASM")
    return qss::FileExtension::QASM;
  if (extStr == "qobj" || extStr == "QOBJ")
    return qss::FileExtension::QOBJ;
  if (extStr == "mlir" || extStr == "MLIR")
    return qss::FileExtension::MLIR;
  if (extStr == "wmem" || extStr == "WMEM")
    return qss::FileExtension::WMEM;
  if (extStr == "qem" || extStr == "QEM")
    return qss::FileExtension::QEM;
  return qss::FileExtension::None;
}

// extracts the file extension and returns the enum qss::FileExtension type
auto getExtension(const std::string &inStr) -> qss::FileExtension {
  auto pos = inStr.find_last_of('.');
  if (pos < inStr.length())
    return strToFileExtension(inStr.substr(pos + 1));
  return qss::FileExtension::None;
}

llvm::Error registerPasses() {
  // TODO: Register standalone passes here.
  llvm::Error err = llvm::Error::success();
  mlir::oq3::registerOQ3Passes();
  mlir::oq3::registerOQ3PassPipeline();
  mlir::qcs::registerQCSPasses();
  mlir::quir::registerQuirPasses();
  mlir::quir::registerQuirPassPipeline();
  mlir::pulse::registerPulsePasses();
  mlir::pulse::registerPulsePassPipeline();
  mlir::registerConversionPasses();

  err = llvm::joinErrors(std::move(err), qssc::hal::registerTargetPasses());
  err = llvm::joinErrors(std::move(err), qssc::hal::registerTargetPipelines());

  mlir::registerAllPasses();
  return err;
}

auto registerPassManagerCLOpts() {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
}

llvm::Error determineInputType() {
  if (inputType == InputType::NONE) {
    inputType = fileExtensionToInputType(getExtension(inputSource));
    if (inputType == InputType::NONE) {
      if (directInput) {
        inputType = InputType::QASM;
      } else {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "Unable to autodetect file extension type! Please specify the "
            "input type with -X");
      }
    } else if (directInput && inputType != InputType::QASM) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "Can only compile direct input when the input type is QASM");
    }
  }
  return llvm::Error::success();
}

void determineOutputType() {
  if (outputFilename != "-") {
    Action extensionAction =
        fileExtensionToAction(getExtension(outputFilename));
    if (extensionAction == Action::None && emitAction == Action::None) {
      llvm::errs()
          << "Can't figure out file extension of specified output file "
          << outputFilename << " defaulting to dumping MLIR\n";
      emitAction = Action::DumpMLIR;
    } else if (emitAction == Action::None) {
      emitAction = extensionAction;
    } else if (extensionAction != emitAction) {
      llvm::errs() << "Warning! The output type in the file extension doesn't "
                      "match the output type specified by --emit!\n";
    }
  } else {
    if (emitAction == Action::None)
      emitAction = Action::DumpMLIR;
  }
}

static void printVersion(llvm::raw_ostream &out) {
  out << "Quantum System Software (QSS) compiler version "
      << qssc::getQSSCVersion() << "\n";
}

/// @brief Build the QSSConfig using the standard sources and assign to the
/// supplied context.
///
/// The configuration precedence order is
/// 1. Default values
/// 2. Environment variables
/// 3. CLI arguments.
///
/// @param context The context to build and register the configuration for.
/// @return The constructed configuration that has been registered for the
/// supplied context.
static llvm::Expected<const qssc::config::QSSConfig &>
buildConfig_(mlir::MLIRContext *context) {
  // First populate the configuration from default values then
  // environment variables.
  auto config = qssc::config::EnvVarConfigBuilder().buildConfig();
  if (auto err = config.takeError())
    // Explicit move required for some systems as automatic move
    // is not recognized.
    return std::move(err);

  // Apply CLI options of top of the configuration constructed above.
  if (auto err = qssc::config::CLIConfigBuilder().populateConfig(*config))
    // Explicit move required for some systems as automatic move
    // is not recognized.
    return std::move(err);

  // Set this as the configuration for the current context
  qssc::config::setContextConfig(context, std::move(*config));

  // Return a constant reference to the managed configuration
  return qssc::config::getContextConfig(context);
}

/// @brief Emit the registered dialects to llvm::outs
static void showDialects_(const DialectRegistry &registry) {
  llvm::outs() << "Registered Dialects:\n";
  for (const auto &registeredDialect : registry.getDialectNames())
    llvm::outs() << registeredDialect << "\n";
}

/// @brief Emit the registered targets to llvm::outs
static void showTargets_() {
  llvm::outs() << "Registered Targets:\n";
  for (const auto &target :
       qssc::hal::registry::TargetSystemRegistry::registeredPlugins()) {
    // Constants chosen empirically to align with --help.
    // TODO: Select constants more intelligently.
    qssc::plugin::registry::printHelpStr(target.second, 2, 57);
  }
}

/// @brief Emit the registered payload to llvm::outs
static void showPayloads_() {
  llvm::outs() << "Registered Payloads:\n";
  for (const auto &payload :
       qssc::payload::registry::PayloadRegistry::registeredPlugins()) {
    // Constants chosen empirically to align with --help.
    // TODO: Select constants more intelligently.
    qssc::plugin::registry::printHelpStr(payload.second, 2, 57);
  }
}

/// @brief Build the target for this MLIRContext based on the supplied config.
/// @param context The supplied context to build the target for.
/// @param config The configuration defining the context to build.
/// @return The constructed TargetSystem.
static llvm::Expected<qssc::hal::TargetSystem &>
buildTarget_(MLIRContext *context, const qssc::config::QSSConfig &config) {
  const auto &targetName = config.targetName;
  const auto &targetConfigPath = config.targetConfigPath;

  if (targetName.has_value()) {
    if (!qssc::hal::registry::TargetSystemRegistry::pluginExists(*targetName))
      // Make sure target exists if specified
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Error: Target " + *targetName +
                                         " is not registered.");
    if (!targetConfigPath.has_value())
      // If the target exists we must have a configuration path.
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "Error: A target configuration path was not specified.");
  }
  qssc::hal::registry::TargetSystemInfo &targetInfo =
      *qssc::hal::registry::TargetSystemRegistry::lookupPluginInfo(
           targetName.value_or(""))
           .getValueOr(qssc::hal::registry::TargetSystemRegistry::
                           nullTargetSystemInfo());

  llvm::Optional<llvm::StringRef> conf{};
  if (targetConfigPath.has_value())
    conf.emplace(*targetConfigPath);

  auto created = targetInfo.createTarget(context, conf);
  if (auto err = created.takeError()) {
    return llvm::joinErrors(
        llvm::createStringError(llvm::inconvertibleErrorCode(),
                                "Unable to create target!"),
        std::move(err));
  }

  return *created.get();
}

/// @brief Generate the final QEM.
/// @param target Target to build the QEM for.
/// @param payload The payload to populate
/// @param moduleOp The module to build for
/// @param ostream The output ostream to populate
/// @return The output error if one occurred.
static llvm::Error generateQEM_(qssc::hal::TargetSystem &target,
                                std::unique_ptr<qssc::payload::Payload> payload,
                                mlir::ModuleOp moduleOp,
                                llvm::raw_ostream *ostream) {
  if (auto err = target.addToPayload(moduleOp, *payload))
    return err;

  if (plaintextPayload)
    payload->writePlain(*ostream);
  else
    payload->write(*ostream);

  return llvm::Error::success();
}

/// @brief Print the output to an ostream.
/// @param ostream The ostream to populate.
/// @param moduleOp The ModuleOp to dump.
static void dumpMLIR_(llvm::raw_ostream *ostream, mlir::ModuleOp moduleOp) {
  moduleOp.print(*ostream);
  *ostream << '\n';
}

static llvm::Error
compile_(int argc, char const **argv, std::string *outputString,
         std::optional<qssc::DiagnosticCallback> diagnosticCb) {
  // Initialize LLVM to start.
  llvm::InitLLVM y(argc, argv);

  // Register the standard passes with MLIR.
  // Must precede the command line parsing.
  if (auto err = registerPasses())
    return err;

  // The MLIR context for this compilation event.
  MLIRContext context{};

  // Pass manager for the compilation
  mlir::PassManager pm(&context);

  // Register the standard dialects with MLIR and prepare a registry and pass
  // pipeline
  DialectRegistry registry = qssc::dialect::registerDialects();

  // Parse the command line options.
  mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");
  registerPassManagerCLOpts();
  llvm::cl::SetVersionPrinter(&printVersion);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Quantum System Software (QSS) Backend Compiler\n");

  mlir::applyPassManagerCLOptions(pm);
  mlir::applyDefaultTimingPassManagerCLOptions(pm);

  // Build the configuration for this compilation event.
  auto configResult = buildConfig_(&context);
  if (auto err = configResult.takeError())
    return err;
  const qssc::config::QSSConfig &config = configResult.get();

  // Populate the context
  context.appendDialectRegistry(registry);
  context.allowUnregisteredDialects(config.allowUnregisteredDialects);
  context.printOpOnDiagnostic(!verifyDiagnostics);

  if (showDialects) {
    showDialects_(registry);
    return llvm::Error::success();
  }

  if (showTargets) {
    showTargets_();
    return llvm::Error::success();
  }

  if (showPayloads) {
    showPayloads_();
    return llvm::Error::success();
  }

  if (showConfig) {
    config.emit(llvm::outs());
    return llvm::Error::success();
  }

  // Build the target for compilation
  auto targetResult = buildTarget_(&context, config);
  if (auto err = targetResult.takeError())
    return err;
  auto &target = targetResult.get();

  if (auto err = determineInputType())
    return err;

  // Set up the input, which is loaded from a file by name by default. With the
  // "--direct" option, the input program can be provided as a string to stdin.
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> file;
  if (!directInput) {
    file = mlir::openInputFile(inputSource, &errorMessage);
    if (!file) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Failed to open input file: " +
                                         errorMessage);
    }
  }

  determineOutputType();

  // Set up the output.
  llvm::raw_ostream *ostream;
  llvm::Optional<llvm::raw_string_ostream> outStringStream;
  auto outputFile = mlir::openOutputFile(outputFilename, &errorMessage);
  std::unique_ptr<qssc::payload::Payload> payload = nullptr;

  if (emitAction == Action::GenQEM) {
    if (outputFilename == "-") {
      auto payloadInfo =
          qssc::payload::registry::PayloadRegistry::lookupPluginInfo("ZIP");
      payload = std::move(
          payloadInfo.getValue()->createPluginInstance(llvm::None).get());
    } else {
      const std::filesystem::path payloadPath(outputFilename.c_str());
      const std::string fNamePrefix = payloadPath.stem();
      const qssc::payload::PayloadConfig config{fNamePrefix, fNamePrefix};
      auto payloadInfo =
          qssc::payload::registry::PayloadRegistry::lookupPluginInfo("ZIP");
      payload =
          std::move(payloadInfo.getValue()->createPluginInstance(config).get());
    }
  }

  if (outputString) {
    outStringStream.emplace(*outputString);
    ostream = outStringStream.getPointer();
  } else {
    if (!outputFile)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Failed to open output file: " +
                                         errorMessage);
    ostream = &outputFile->os();
  }

  mlir::ModuleOp moduleOp;

  if (inputType == InputType::QASM) {
    if (emitAction >= Action::DumpMLIR) {
      moduleOp = mlir::ModuleOp::create(FileLineColLoc::get(
          &context, directInput ? std::string{"-"} : inputSource, 0, 0));
    }

    if (auto frontendError = qssc::frontend::openqasm3::parse(
            inputSource, !directInput, emitAction == Action::DumpAST,
            emitAction == Action::DumpASTPretty, emitAction >= Action::DumpMLIR,
            moduleOp, std::move(diagnosticCb)))
      return frontendError;

    if (emitAction < Action::DumpMLIR)
      return llvm::Error::success();
  } // if input == QASM

  if (inputType == InputType::MLIR) {
    // ------------------------------------------------------------
    // The following section was copied from processBuffer() in:
    //      ../third_party/llvm-project/mlir/lib/Support/MlirOptMain.cpp

    // Tell sourceMgr about this buffer, which is what the parser will pick up.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

    // Parse the input file.
    // ------------------------------------------------------------
    // The following section was copied from performActions() in:
    //      ../third_party/llvm-project/mlir/lib/Support/MlirOptMain.cpp

    // Disable multi-threading when parsing the input file. This removes the
    // unnecessary/costly context synchronization when parsing.
    bool wasThreadingEnabled = context.isMultithreadingEnabled();
    context.disableMultithreading();

    // Parse the input file and reset the context threading state.
    mlir::OwningOpRef<ModuleOp> module(
        mlir::parseSourceFile(sourceMgr, &context));
    context.enableMultithreading(wasThreadingEnabled);
    if (!module)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Problem parsing source file " +
                                         inputSource);
    moduleOp = module.release();
  } // if input == MLIR

  auto errorHandler = [&](const Twine &msg) {
    emitError(UnknownLoc::get(&context)) << msg;
    return failure();
  };

  // at this point we have QUIR+Pulse in the moduleOp from either the
  // QASM/AST or MLIR file

  // Build the provided pipeline.
  if (failed(passPipeline.addToPipeline(pm, errorHandler)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Problem adding passes to passPipeline!");

  if (emitAction > Action::DumpMLIR && config.addTargetPasses)
    // check if the target quir to std pass has been specified in the CL
    if (auto err = target.addPayloadPasses(pm))
      return llvm::joinErrors(
          llvm::createStringError(llvm::inconvertibleErrorCode(),
                                  "Failure while preparing target passes"),
          std::move(err));

  // Run the pipeline.
  if (failed(pm.run(moduleOp)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Problems running the compiler pipeline!");

  if (emitAction == Action::DumpMLIR) {
    // Print the output.
    dumpMLIR_(ostream, moduleOp);
  }

  if (emitAction == Action::GenQEM) {

    if (includeSourceInPayload && directInput) {
      if (inputType == InputType::QASM)
        payload->addFile("manifest/input.qasm", inputSource + "\n");
      else if (inputType == InputType::MLIR)
        payload->addFile("manifest/input.mlir", inputSource + "\n");
    }

    if (auto err = generateQEM_(target, std::move(payload), moduleOp, ostream))
      return err;
  }

  // ------------------------------------------------------------

  // Keep the output if no errors have occurred so far
  if (outputString) {
    outStringStream.getValue().str();
    if (outputFile && outputFilename != "-")
      outputFile->os() << *outputString;
  }
  if (outputFile && outputFilename != "-")
    outputFile->keep();

  return llvm::Error::success();
}

int qssc::compile(int argc, char const **argv, std::string *outputString,
                  std::optional<DiagnosticCallback> diagnosticCb) {
  if (auto err = compile_(argc, argv, outputString, std::move(diagnosticCb))) {
    llvm::logAllUnhandledErrors(std::move(err), llvm::errs(), "Error: ");
    return 1;
  }

  return 0;
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
