//===- api.cpp --------------------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023, 2024.
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

#include "API/errors.h"
#include "Arguments/Arguments.h"
#include "Config/CLIConfig.h"
#include "Config/QSSConfig.h"
#include "Dialect/OQ3/Transforms/Passes.h"
#include "Dialect/Pulse/Transforms/Passes.h"
#include "Dialect/QCS/Utils/ParameterInitialValueAnalysis.h"
#include "Dialect/QUIR/Transforms/Passes.h"
#include "Dialect/RegisterDialects.h"
#include "Dialect/RegisterPasses.h"
#include "Frontend/OpenQASM3/OpenQASM3Frontend.h"
#include "HAL/Compile/TargetCompilationManager.h"
#include "HAL/Compile/ThreadedCompilationManager.h"
#include "HAL/PassRegistration.h"
#include "HAL/TargetSystem.h"
#include "HAL/TargetSystemInfo.h"
#include "HAL/TargetSystemRegistry.h"
#include "Payload/Payload.h"
#include "Payload/PayloadRegistry.h"
#include "Plugin/PluginInfo.h"
#include "QSSC.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Tools/ParseUtilities.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

using namespace mlir;
// <<<<<<< HEAD

// static llvm::cl::opt<std::string> inputSource(
//     llvm::cl::Positional, llvm::cl::desc("Input filename or program source"),
//     llvm::cl::init("-"), llvm::cl::cat(qssc::config::getQSSCCategory()));

// static llvm::cl::opt<std::string>
//     outputFilename("o", llvm::cl::desc("Output filename"),
//                    llvm::cl::value_desc("filename"), llvm::cl::init("-"),
//                    llvm::cl::cat(qssc::config::getQSSCCategory()));

// static llvm::cl::opt<bool>
//     directInput("direct",
//                 llvm::cl::desc("Accept the input program directly as a
//                 string"), llvm::cl::cat(qssc::config::getQSSCCategory()));

// static llvm::cl::opt<bool> verifyDiagnostics(
//     "verify-diagnostics",
//     llvm::cl::desc("Check that emitted diagnostics match "
//                    "expected-* lines on the corresponding line"),
//     llvm::cl::init(false), llvm::cl::cat(qssc::config::getQSSCCategory()));

// #ifndef NOVERIFY
// #define VERIFY_PASSES_DEFAULT true
// #else
// #define VERIFY_PASSES_DEFAULT false
// #endif

// static llvm::cl::opt<bool> verifyPasses(
//     "verify-each",
//     llvm::cl::desc("Run the verifier after each transformation pass"),
//     llvm::cl::init(VERIFY_PASSES_DEFAULT),
//     llvm::cl::cat(qssc::config::getQSSCCategory()));

// static llvm::cl::opt<bool> showDialects(
//     "show-dialects", llvm::cl::desc("Print the list of registered dialects"),
//     llvm::cl::init(false), llvm::cl::cat(qssc::config::getQSSCCategory()));

// static llvm::cl::opt<bool> showTargets(
//     "show-targets", llvm::cl::desc("Print the list of registered targets"),
//     llvm::cl::init(false), llvm::cl::cat(qssc::config::getQSSCCategory()));

// static llvm::cl::opt<bool> showPayloads(
//     "show-payloads", llvm::cl::desc("Print the list of registered payloads"),
//     llvm::cl::init(false), llvm::cl::cat(qssc::config::getQSSCCategory()));

// static llvm::cl::opt<bool> showConfig(
//     "show-config", llvm::cl::desc("Print the loaded compiler
//     configuration."), llvm::cl::init(false),
//     llvm::cl::cat(qssc::config::getQSSCCategory()));

// static llvm::cl::opt<bool> plaintextPayload(
//     "plaintext-payload", llvm::cl::desc("Write the payload in plaintext"),
//     llvm::cl::init(false), llvm::cl::cat(qssc::config::getQSSCCategory()));

// static llvm::cl::opt<bool> includeSourceInPayload(
//     "include-source", llvm::cl::desc("Write the input source into the
//     payload"), llvm::cl::init(false),
//     llvm::cl::cat(qssc::config::getQSSCCategory()));

// static llvm::cl::opt<bool>
//     bypassPipeline("bypass-pipeline", llvm::cl::desc("Bypass the pipeline"),
//                    llvm::cl::init(false),
//                    llvm::cl::cat(qssc::config::getQSSCCategory()));

// namespace {
// enum InputType { NONE, QASM, MLIR, QOBJ };
// } // anonymous namespace
// static llvm::cl::opt<enum InputType> inputType(
//     "X", llvm::cl::init(InputType::NONE),
//     llvm::cl::desc("Specify the kind of input desired"),
//     llvm::cl::values(
//         clEnumValN(InputType::QASM, "qasm",
//                    "load the input file as an OpenQASM 3.0 source")),
//     llvm::cl::values(clEnumValN(MLIR, "mlir",
//                                 "load the input file as an MLIR file")),
//     llvm::cl::values(clEnumValN(QOBJ, "qobj",
//                                 "load the input file as a QOBJ file")));

// static llvm::cl::opt<enum Action> emitAction(
//     "emit", llvm::cl::init(Action::None),
//     llvm::cl::desc("Select the kind of output desired"),
//     llvm::cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
//     llvm::cl::values(clEnumValN(DumpASTPretty, "ast-pretty",
//                                 "pretty print the AST")),
//     llvm::cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
//     llvm::cl::values(clEnumValN(DumpWaveMem, "wavemem",
//                                 "output the waveform memory")),
//     llvm::cl::values(clEnumValN(GenQEM, "qem",
//                                 "generate a quantum executable module (qem) "
//                                 "for execution on hardware")),
//     llvm::cl::values(clEnumValN(
//         GenQEQEM, "qe-qem",
//         "generate a target-specific quantum executable module (qeqem) "
//         "for execution on hardware")));

// namespace qss {
// enum FileExtension { None, AST, ASTPRETTY, QASM, QOBJ, MLIR, WMEM, QEM, QEQEM
// }; } // namespace qss

// auto fileExtensionToStr(const qss::FileExtension &inExt) -> std::string {
//   switch (inExt) {
//   case qss::FileExtension::AST:
//     return "ast";
//     break;
//   case qss::FileExtension::ASTPRETTY:
//     return "ast-pretty";
//     break;
//   case qss::FileExtension::QASM:
//     return "qasm";
//     break;
//   case qss::FileExtension::QOBJ:
//     return "qobj";
//     break;
//   case qss::FileExtension::MLIR:
//     return "mlir";
//     break;
//   case qss::FileExtension::WMEM:
//     return "wmem";
//     break;
//   case qss::FileExtension::QEM:
//     return "qem";
//     break;
//   case qss::FileExtension::QEQEM:
//     return "qeqem";
//     break;
//   default:
//     return "none";
//     break;
//   }
//   return "none";
// }

// auto fileExtensionToInputType(const qss::FileExtension &inExt) -> InputType {
//   switch (inExt) {
//   case qss::FileExtension::QASM:
//     return InputType::QASM;
//     break;
//   case qss::FileExtension::QOBJ:
//     return InputType::QOBJ;
//     break;
//   case qss::FileExtension::MLIR:
//     return InputType::MLIR;
//     break;
//   default:
//     break;
//   }
//   return InputType::NONE;
// }

// auto fileExtensionToAction(const qss::FileExtension &inExt) -> Action {
//   switch (inExt) {
//   case qss::FileExtension::AST:
//     return Action::DumpAST;
//     break;
//   case qss::FileExtension::ASTPRETTY:
//     return Action::DumpASTPretty;
//     break;
//   case qss::FileExtension::MLIR:
//     return Action::DumpMLIR;
//     break;
//   case qss::FileExtension::WMEM:
//     return Action::DumpWaveMem;
//     break;
//   case qss::FileExtension::QEM:
//     return Action::GenQEM;
//     break;
//   case qss::FileExtension::QEQEM:
//     return Action::GenQEQEM;
//     break;
//   default:
//     break;
//   }
//   return Action::None;
// }

// auto strToFileExtension(const std::string &extStr) -> qss::FileExtension {
//   if (extStr == "ast" || extStr == "AST")
//     return qss::FileExtension::AST;
//   if (extStr == "ast-pretty" || extStr == "AST-PRETTY")
//     return qss::FileExtension::ASTPRETTY;
//   if (extStr == "qasm" || extStr == "QASM")
//     return qss::FileExtension::QASM;
//   if (extStr == "qobj" || extStr == "QOBJ")
//     return qss::FileExtension::QOBJ;
//   if (extStr == "mlir" || extStr == "MLIR")
//     return qss::FileExtension::MLIR;
//   if (extStr == "wmem" || extStr == "WMEM")
//     return qss::FileExtension::WMEM;
//   if (extStr == "qem" || extStr == "QEM")
//     return qss::FileExtension::QEM;
//   if (extStr == "qeqem" || extStr == "QEQEM")
//     return qss::FileExtension::QEQEM;
//   return qss::FileExtension::None;
// }

// // extracts the file extension and returns the enum qss::FileExtension type
// auto getExtension(const std::string &inStr) -> qss::FileExtension {
//   auto pos = inStr.find_last_of('.');
//   if (pos < inStr.length())
//     return strToFileExtension(inStr.substr(pos + 1));
//   return qss::FileExtension::None;
// }
using namespace qssc::config;

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

auto registerCLOpts() {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  qssc::hal::compile::registerTargetCompilationManagerCLOptions();
}

namespace {
void printVersion(llvm::raw_ostream &out) {
  out << "Quantum System Software (QSS) compiler version "
      << qssc::getQSSCVersion() << "\n";
}

/// @brief Emit the registered dialects to llvm::outs
void showDialects_(const mlir::DialectRegistry &registry) {
  llvm::outs() << "Registered Dialects:\n";
  for (const auto &registeredDialect : registry.getDialectNames())
    llvm::outs() << registeredDialect << "\n";
}

/// @brief Emit the registered targets to llvm::outs
void showTargets_() {
  llvm::outs() << "Registered Targets:\n";
  for (const auto &target :
       qssc::hal::registry::TargetSystemRegistry::registeredPlugins()) {
    // Constants chosen empirically to align with --help.
    // TODO: Select constants more intelligently.
    qssc::plugin::registry::printHelpStr(target.second, 2, 57);
  }
}

/// @brief Emit the registered payload to llvm::outs
void showPayloads_() {
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
llvm::Expected<qssc::hal::TargetSystem &>
buildTarget_(MLIRContext *context, const qssc::config::QSSConfig &config,
             mlir::TimingScope &timing) {

  mlir::TimingScope const buildTargetTiming = timing.nest("build-target");

  const auto &targetName = config.getTargetName();
  const auto &targetConfigPath = config.getTargetConfigPath();

  if (targetName.has_value()) {
    if (!qssc::hal::registry::TargetSystemRegistry::pluginExists(*targetName))
      // Make sure target exists if specified
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Target " + *targetName +
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
           .value_or(qssc::hal::registry::TargetSystemRegistry::
                         nullTargetSystemInfo());

  std::optional<llvm::StringRef> conf{};
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
/// @param targetCompilationManager Target compilation to build the QEM with.
/// @param payload The payload to populate
/// @param moduleOp The module to build for
/// @param ostream The output ostream to populate
/// @return The output error if one occurred.
llvm::Error generateQEM_(
    const QSSConfig &config,
    qssc::hal::compile::TargetCompilationManager *targetCompilationManager,
    std::unique_ptr<qssc::payload::Payload> payload, mlir::ModuleOp moduleOp,
    llvm::raw_ostream *ostream, mlir::TimingScope &timing) {

  mlir::TimingScope buildQEMTiming = timing.nest("build-qem");
  targetCompilationManager->enableTiming(buildQEMTiming);
  if (auto err = targetCompilationManager->compilePayload(
          moduleOp, *payload,
          /* doCompileMLIR=*/!config.shouldBypassPayloadTargetCompilation()))
    return err;
  targetCompilationManager->disableTiming();

  mlir::TimingScope const writePayloadTiming =
      buildQEMTiming.nest("write-payload");
  if (config.shouldEmitPlaintextPayload())
    payload->writePlain(*ostream);
  else
    payload->write(*ostream);

  return llvm::Error::success();
}

/// @brief Print the output to an ostream.
/// @param ostream The ostream to populate.
/// @param moduleOp The ModuleOp to dump.
void dumpMLIR_(llvm::raw_ostream *ostream, mlir::ModuleOp moduleOp) {
  moduleOp.print(*ostream);
  *ostream << '\n';
}

using ErrorHandler = function_ref<LogicalResult(const Twine &)>;

llvm::Error buildPassManager_(mlir::PassManager &pm, bool verifyPasses) {
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm)))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Unable to apply pass manager command line options");

  // Configure verifier
  pm.enableVerifier(verifyPasses);

  return llvm::Error::success();
}

llvm::Error buildPassManager(const QSSConfig &config, mlir::PassManager &pm,
                             ErrorHandler errorHandler, bool verifyPasses,
                             mlir::TimingScope &timing) {
  if (auto err = buildPassManager_(pm, verifyPasses))
    return err;

  pm.enableTiming(timing);

  // Build the provided pipeline.
  if (failed(config.setupPassPipeline(pm)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Problem adding passes to passPipeline!");
  return llvm::Error::success();
}

/// @brief Emit MLIR from the compiler
/// @param ostream Output stream to emit to
/// @param context The active MLIR context
/// @param moduleOp The module operation to process and emit
/// @param config Compilation configuration options
/// @param targetCompilationManager The target's compilation scheduler
/// @param errorHandler MLIR error handler
/// @return
llvm::Error emitMLIR_(
    llvm::raw_ostream *ostream, mlir::MLIRContext &context,
    mlir::ModuleOp moduleOp, const QSSConfig &config,
    qssc::hal::compile::ThreadedCompilationManager &targetCompilationManager,
    ErrorHandler errorHandler, mlir::TimingScope &timing) {

  mlir::TimingScope emitMlirTiming = timing.nest("emit-mlir");

  if (config.shouldCompileTargetIR()) {
    // Check if we can run the target compilation scheduler.
    if (config.shouldAddTargetPasses()) {
      targetCompilationManager.enableTiming(emitMlirTiming);
      if (auto err = targetCompilationManager.compileMLIR(moduleOp))
        return llvm::joinErrors(
            llvm::createStringError(llvm::inconvertibleErrorCode(),
                                    "Failure while preparing target passes"),
            std::move(err));
      targetCompilationManager.disableTiming();
    }
  }

  // Print the output.
  dumpMLIR_(ostream, moduleOp);

  return llvm::Error::success();
}

/// @brief Emit a QEM payload from the compiler
/// @param ostream Output stream to emit to
/// @param payload The payload to emit
/// @param moduleOp The module operation to process and emit
/// @param targetCompilationManager The target's compilation scheduler
/// @return
llvm::Error emitQEM_(
    const QSSConfig &config, llvm::raw_ostream *ostream,
    std::unique_ptr<qssc::payload::Payload> payload, mlir::ModuleOp moduleOp,
    qssc::hal::compile::ThreadedCompilationManager &targetCompilationManager,
    mlir::TimingScope &timing) {
  if (config.shouldIncludeSource()) {
    if (config.isDirectInput()) {
      if (config.getInputType() == InputType::QASM)
        payload->addFile("manifest/input.qasm",
                         (config.getInputSource() + "\n").str());
      else if (config.getInputType() == InputType::MLIR)
        payload->addFile("manifest/input.mlir",
                         (config.getInputSource() + "\n").str());
      else
        llvm_unreachable("Unhandled input file type");
    } else { // just copy the input file
      std::ifstream fileStream(config.getInputSource().str());
      std::stringstream fileSS;
      fileSS << fileStream.rdbuf();

      if (config.getInputType() == InputType::QASM)
        payload->addFile("manifest/input.qasm", fileSS.str());
      else if (config.getInputType() == InputType::MLIR)
        payload->addFile("manifest/input.mlir", fileSS.str());
      else
        llvm_unreachable("Unhandled input file type");

      fileStream.close();
    }
  }

  if (auto err = generateQEM_(config, &targetCompilationManager,
                              std::move(payload), moduleOp, ostream, timing))
    return err;

  return llvm::Error::success();
}

/// @brief Handler for the Diagnostic Engine.
///
///        Uses qssc::emitDiagnostic to forward diagnostic to the python
///        diagnostic callback.
///        Prints diagnostic to llvm::errs to mimic default handler.
///  @param diagnostic MLIR diagnostic from the Diagnostic Engine
///  @param diagnosticCb Handle to python diagnostic callback
void diagEngineHandler(mlir::Diagnostic &diagnostic,
                       std::optional<qssc::DiagnosticCallback> diagnosticCb) {

  // map diagnostic severity to qssc severity
  auto severity = diagnostic.getSeverity();
  qssc::Severity qssc_severity = qssc::Severity::Error;
  switch (severity) {
  case mlir::DiagnosticSeverity::Error:
    qssc_severity = qssc::Severity::Error;
    break;
  case mlir::DiagnosticSeverity::Warning:
    qssc_severity = qssc::Severity::Warning;
    break;
  case mlir::DiagnosticSeverity::Note:
  case mlir::DiagnosticSeverity::Remark:
    qssc_severity = qssc::Severity::Info;
  }
  // emit diagnostic cast to void to discard result as it is not needed here
  if (qssc_severity == qssc::Severity::Error) {
    (void)qssc::emitDiagnostic(std::move(diagnosticCb), qssc_severity,
                               qssc::ErrorCategory::QSSCompilationFailure,
                               diagnostic.str());
  }

  // emit to llvm::errs as well to mimic default handler
  diagnostic.getLocation().print(llvm::errs());
  llvm::errs() << ": ";
  // based on mlir's Diagnostic.cpp:getDiagKindStr which is static
  switch (severity) {
  case mlir::DiagnosticSeverity::Note:
    llvm::errs() << "note: ";
    break;
  case mlir::DiagnosticSeverity::Warning:
    llvm::errs() << "warning: ";
    break;
  case mlir::DiagnosticSeverity::Error:
    llvm::errs() << "error: ";
    break;
  case mlir::DiagnosticSeverity::Remark:
    llvm::errs() << "remark: ";
  }
  llvm::errs() << diagnostic << "\n";
  return;
}

llvm::Error compile_(int argc, char const **argv, std::string *outputString,
                     std::optional<qssc::DiagnosticCallback> diagnosticCb) {

  // Initialize LLVM to start.
  llvm::InitLLVM const y(argc, argv);

  // Register the standard passes with MLIR.
  // Must precede the command line parsing.
  if (auto err = qssc::dialect::registerPasses())
    return err;

  // Register the standard dialects with MLIR and prepare a registry and pass
  // pipeline
  mlir::DialectRegistry registry;
  qssc::dialect::registerDialects(registry);

  // Register all extensions
  mlir::registerAllExtensions(registry);

  registerCLOpts();
  // Register CL config builder prior to parsing
  CLIConfigBuilder::registerCLOptions(registry);
  llvm::cl::SetVersionPrinter(&printVersion);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Quantum System Software (QSS) Backend Compiler\n");

  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  TimingScope timing = tm.getRootScope();

  // The MLIR context for this compilation event.
  // Instantiate after parsing command line options.
  MLIRContext context{};

  mlir::TimingScope buildConfigTiming = timing.nest("build-config");
  auto configResult = qssc::config::buildToolConfig();
  if (auto err = configResult.takeError())
    return err;
  qssc::config::QSSConfig const config = configResult.get();
  qssc::config::setContextConfig(&context, config);
  buildConfigTiming.stop();

  // Populate the context
  context.appendDialectRegistry(registry);
  context.allowUnregisteredDialects(config.shouldAllowUnregisteredDialects());
  context.printOpOnDiagnostic(!config.shouldVerifyDiagnostics());

  // Register LLVM dialect and all infrastructure required for translation to
  // LLVM IR
  mlir::registerBuiltinDialectTranslation(context);
  mlir::registerLLVMDialectTranslation(context);

  if (config.shouldShowDialects()) {
    showDialects_(registry);
    return llvm::Error::success();
  }

  if (config.shouldShowTargets()) {
    showTargets_();
    return llvm::Error::success();
  }

  if (config.shouldShowPayloads()) {
    showPayloads_();
    return llvm::Error::success();
  }

  if (config.shouldShowConfig()) {
    config.emit(llvm::outs());
    return llvm::Error::success();
  }

  // Build the target for compilation
  auto targetResult = buildTarget_(&context, config, timing);
  if (auto err = targetResult.takeError())
    return err;
  auto &target = targetResult.get();

  // Set up the input, which is loaded from a file by name by default. With the
  // "--direct" option, the input program can be provided as a string to stdin.
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> file;
  if (!config.isDirectInput()) {
    file = mlir::openInputFile(config.getInputSource(), &errorMessage);
    if (!file) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Failed to open input file: " +
                                         errorMessage);
    }
  } else {
    file = llvm::MemoryBuffer::getMemBuffer(config.getInputSource(),
                                            /*bufferName=*/"direct");
  }

  context.getDiagEngine().registerHandler([&](mlir::Diagnostic &diagnostic) {
    diagEngineHandler(diagnostic, diagnosticCb);
  });

  // Set up the output.
  llvm::raw_ostream *ostream;
  std::optional<llvm::raw_string_ostream> outStringStream;
  auto outputFile =
      mlir::openOutputFile(config.getOutputFilePath(), &errorMessage);
  std::unique_ptr<qssc::payload::Payload> payload = nullptr;

  if (config.getEmitAction() == EmitAction::QEQEM &&
      !config.getTargetName().has_value())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Unsupported target-specific payload: no target");
  if (config.getEmitAction() == EmitAction::QEM ||
      config.getEmitAction() == EmitAction::QEQEM) {
    const std::filesystem::path payloadPath(config.getOutputFilePath().str());
    const std::string fNamePrefix = payloadPath.stem();
    const auto payloadName = (config.getEmitAction() == EmitAction::QEM)
                                 ? "ZIP"
                                 : config.getTargetName().value();
    auto payloadInfo =
        qssc::payload::registry::PayloadRegistry::lookupPluginInfo(payloadName);
    if (payloadInfo == std::nullopt)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Unsupported target-specific payload: " +
                                         payloadName);
    if (config.getOutputFilePath() == "-") {
      payload = std::move(
          payloadInfo.value()->createPluginInstance(std::nullopt).get());
    } else {
      const qssc::payload::PayloadConfig payloadConfig{
          fNamePrefix, fNamePrefix, config.getVerbosityLevel()};
      payload = std::move(
          payloadInfo.value()->createPluginInstance(payloadConfig).get());
    }
  }

  if (outputString) {
    outStringStream.emplace(*outputString);
    if (outStringStream.has_value())
      ostream = std::addressof(*outStringStream);
  } else {
    if (!outputFile)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Failed to open output file: " +
                                         errorMessage);
    ostream = &outputFile->os();
  }

  mlir::ModuleOp moduleOp;

  if (config.getInputType() == InputType::QASM) {

    mlir::TimingScope loadQASM3Timing = timing.nest("load-qasm3");

    if (config.getEmitAction() >= EmitAction::MLIR) {
      moduleOp = mlir::ModuleOp::create(FileLineColLoc::get(
          &context,
          config.isDirectInput() ? std::string{"-"} : config.getInputSource(),
          0, 0));
    }

    if (auto frontendError = qssc::frontend::openqasm3::parse(
            config.getInputSource().str(), !config.isDirectInput(),
            config.getEmitAction() == EmitAction::AST,
            config.getEmitAction() == EmitAction::ASTPretty,
            config.getEmitAction() >= EmitAction::MLIR, moduleOp, diagnosticCb,
            loadQASM3Timing))
      return frontendError;

    if (config.getEmitAction() < EmitAction::MLIR)
      return llvm::Error::success();
  } // if input == QASM

  if (config.getInputType() == InputType::MLIR) {

    mlir::TimingScope mlirParserTiming = timing.nest("parse-mlir");

    // Tell sourceMgr about this buffer, which is what the parser will pick up.
    auto sourceMgr = std::make_shared<llvm::SourceMgr>();
    sourceMgr->AddNewSourceBuffer(std::move(file), llvm::SMLoc());

    // Implemented following -
    // https://github.com/llvm/llvm-project/blob/llvmorg-17.0.6/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp#L333-L362

    // Disable multi-threading when parsing the input file. This removes the
    // unnecessary/costly context synchronization when parsing.
    const bool wasThreadingEnabled = context.isMultithreadingEnabled();
    context.disableMultithreading();

    // Prepare the parser config, and attach any useful/necessary resource
    // handlers. Unhandled external resources are treated as passthrough, i.e.
    // they are not processed and will be emitted directly to the output
    // untouched.
    mlir::PassReproducerOptions reproOptions;
    mlir::FallbackAsmResourceMap fallbackResourceMap;
    mlir::ParserConfig parseConfig(&context, /*verifyAfterParse=*/true,
                                   &fallbackResourceMap);
    if (config.shouldRunReproducer())
      reproOptions.attachResourceParser(parseConfig);

    // Parse the input file and reset the context threading state.
    mlir::OwningOpRef<Operation *> op = mlir::parseSourceFileForTool(
        sourceMgr, parseConfig, !config.shouldUseExplicitModule());

    mlirParserTiming.stop();

    if (!op)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Problem parsing source file " +
                                         config.getInputSource());

    // Cannot currently perform round-trip verification as
    // doVerificationRoundTrip is not part of MLIR's public
    // API -
    // https://github.com/llvm/llvm-project/blob/llvmorg-17.0.6/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp#L250
    if (config.shouldVerifyRoundtrip())
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "The qss-compiler does not currently support roundtrip verification. "
          "Please use the qss-opt tool instead.");

    context.enableMultithreading(wasThreadingEnabled);

    moduleOp = mlir::dyn_cast<mlir::ModuleOp>(op.release());
  } // if input == MLIR

  auto errorHandler = [&](const Twine &msg) {
    // format msg to python handler as a compilation failure
    (void)qssc::emitDiagnostic(diagnosticCb, qssc::Severity::Error,
                               qssc::ErrorCategory::QSSCompilationFailure,
                               msg.str());
    emitError(UnknownLoc::get(&context)) << msg;
    return mlir::failure();
  };

  // at this point we have QUIR+Pulse in the moduleOp from either the
  // QASM/AST or MLIR file

  bool verifyPasses = config.shouldVerifyPasses();

  auto targetCompilationManager =
      qssc::hal::compile::ThreadedCompilationManager(
          target, &context, [&](mlir::PassManager &pm) -> llvm::Error {
            if (auto err = buildPassManager_(pm, verifyPasses))
              return err;
            return llvm::Error::success();
          });
  if (mlir::failed(qssc::hal::compile::applyTargetCompilationManagerCLOptions(
          targetCompilationManager)))
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Unable to apply target compilation options.");

  // Run additional passes specified on the command line

  mlir::TimingScope commandLinePassesTiming =
      timing.nest("command-line-passes");
  mlir::PassManager pm(&context);
  if (auto err = buildPassManager(config, pm, errorHandler, verifyPasses,
                                  commandLinePassesTiming))
    return err;

  if (pm.size() && failed(pm.run(moduleOp)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Problems running the compiler pipeline!");
  commandLinePassesTiming.stop();

  // Prepare outputs
  if (config.getEmitAction() == EmitAction::MLIR) {
    if (auto err = emitMLIR_(ostream, context, moduleOp, config,
                             targetCompilationManager, errorHandler, timing))
      return err;
  }

  if (config.getEmitAction() == EmitAction::QEM ||
      config.getEmitAction() == EmitAction::QEQEM) {
    if (auto err = emitQEM_(config, ostream, std::move(payload), moduleOp,
                            targetCompilationManager, timing))
      return err;
  }

  // ------------------------------------------------------------

  // Keep the output if no errors have occurred so far
  if (outputString) {
    if (outputFile && config.getOutputFilePath() != "-")
      outputFile->os() << *outputString;
  }
  if (outputFile && config.getOutputFilePath() != "-")
    outputFile->keep();

  return llvm::Error::success();
}
} // anonymous namespace

int qssc::compile(int argc, char const **argv, std::string *outputString,
                  std::optional<DiagnosticCallback> diagnosticCb) {
  if (auto err = compile_(argc, argv, outputString, std::move(diagnosticCb))) {
    llvm::logAllUnhandledErrors(std::move(err), llvm::errs(), "Error: ");
    return 1;
  }

  return 0;
}
