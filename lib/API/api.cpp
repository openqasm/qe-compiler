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
#include "Dialect/RegisterDialects.h"
#include "Dialect/RegisterPasses.h"
#include "Frontend/OpenQASM3/OpenQASM3Frontend.h"
#include "HAL/Compile/TargetCompilationManager.h"
#include "HAL/Compile/ThreadedCompilationManager.h"
#include "HAL/TargetSystem.h"
#include "HAL/TargetSystemInfo.h"
#include "HAL/TargetSystemRegistry.h"
#include "Payload/Payload.h"
#include "Payload/PayloadRegistry.h"
#include "Plugin/PluginInfo.h"
#include "QSSC.h"

#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllExtensions.h"
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
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <memory>
#include <optional>
#include <stdio.h> // NOLINT: fileno is not in cstdio as suggested
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>

using namespace mlir;
using namespace qssc::config;

//===--------------------------- Compilation ----------------------------===//

namespace {

auto registerCLOpts() {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
  qssc::hal::compile::registerTargetCompilationManagerCLOptions();
}

void printVersion(llvm::raw_ostream &out) {
  out << "Quantum System Software (QSS) compiler version "
      << qssc::getQSSCVersion() << "\n";
}

/// @brief Emit the registered dialects to llvm::outs
void showDialects(const mlir::DialectRegistry &registry) {
  llvm::outs() << "Registered Dialects:\n";
  for (const auto &registeredDialect : registry.getDialectNames())
    llvm::outs() << registeredDialect << "\n";
}

/// @brief Emit the registered targets to llvm::outs
void showTargets() {
  llvm::outs() << "Registered Targets:\n";
  for (const auto &target :
       qssc::hal::registry::TargetSystemRegistry::registeredPlugins()) {
    // Constants chosen empirically to align with --help.
    // TODO: Select constants more intelligently.
    qssc::plugin::registry::printHelpStr(target.second, 2, 57);
  }
}

/// @brief Emit the registered payload to llvm::outs
void showPayloads() {
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
buildTarget(MLIRContext *context, const qssc::config::QSSConfig &config,
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
llvm::Error generateQEM(
    const QSSConfig &config,
    qssc::hal::compile::TargetCompilationManager *targetCompilationManager,
    std::unique_ptr<qssc::payload::Payload> payload, mlir::ModuleOp moduleOp,
    llvm::raw_ostream &ostream, mlir::TimingScope &timing) {

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
    payload->writePlain(ostream);
  else
    payload->write(ostream);

  return llvm::Error::success();
}

/// @brief Print the MLIR output to an ostream.
void dumpMLIR(llvm::raw_ostream &ostream, mlir::ModuleOp moduleOp) {
  moduleOp.print(ostream);
  ostream << '\n';
}

/// Emit MLIR bytecode to an ostream.
llvm::Error dumpBytecode(const QSSConfig &config, llvm::raw_ostream &os,
                         mlir::ModuleOp moduleOp,
                         mlir::FallbackAsmResourceMap &fallbackResourceMap) {

  BytecodeWriterConfig writerConfig(fallbackResourceMap);
  if (auto v = config.bytecodeVersionToEmit())
    writerConfig.setDesiredBytecodeVersion(*v);

  if (mlir::failed(mlir::writeBytecodeToFile(moduleOp, os, writerConfig)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Unable to emit module bytecode");

  return llvm::Error::success();
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
llvm::Error emitMLIR(
    llvm::raw_ostream &ostream, mlir::MLIRContext &context,
    mlir::ModuleOp moduleOp, mlir::FallbackAsmResourceMap &fallbackResourceMap,
    const QSSConfig &config,
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
  if (config.getEmitAction() == EmitAction::MLIR)
    dumpMLIR(ostream, moduleOp);
  else if (config.getEmitAction() == EmitAction::Bytecode)
    if (auto err = dumpBytecode(config, ostream, moduleOp, fallbackResourceMap))
      return err;

  return llvm::Error::success();
}

/// @brief Emit a QEM payload from the compiler
/// @param ostream Output stream to emit to
/// @param payload The payload to emit
/// @param moduleOp The module operation to process and emit
/// @param targetCompilationManager The target's compilation scheduler
/// @return
llvm::Error emitQEM(
    const QSSConfig &config, llvm::raw_ostream &ostream,
    std::unique_ptr<qssc::payload::Payload> payload, mlir::ModuleOp moduleOp,
    qssc::hal::compile::ThreadedCompilationManager &targetCompilationManager,
    const llvm::MemoryBuffer *sourceBuffer, mlir::TimingScope &timing) {
  if (config.shouldIncludeSource()) {
    if (config.getInputType() != InputType::Undetected)
      payload->addFile("manifest/input." + to_string(inputTypeToFileExtension(
                                               config.getInputType())),
                       sourceBuffer->getBuffer());
    else
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "The input source type does not support embedding in the payload");
  }

  if (auto err = generateQEM(config, &targetCompilationManager,
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
                       const qssc::OptDiagnosticCallback &diagnosticCb) {

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
    (void)qssc::emitDiagnostic(diagnosticCb, qssc_severity,
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

llvm::Error applyEmitAction(
    const QSSConfig &config, llvm::raw_ostream &outputStream,
    std::unique_ptr<qssc::payload::Payload> payload, mlir::MLIRContext &context,
    mlir::ModuleOp moduleOp, const llvm::MemoryBuffer *sourceBuffer,
    mlir::FallbackAsmResourceMap &fallbackResourceMap,
    qssc::hal::compile::ThreadedCompilationManager &targetCompilationManager,
    ErrorHandler errorHandler, mlir::TimingScope &timing) {
  // Prepare outputs
  if (config.getEmitAction() == EmitAction::MLIR ||
      config.getEmitAction() == EmitAction::Bytecode) {
    if (auto err =
            emitMLIR(outputStream, context, moduleOp, fallbackResourceMap,
                     config, targetCompilationManager, errorHandler, timing))
      return err;
  }

  if (config.getEmitAction() == EmitAction::QEM ||
      config.getEmitAction() == EmitAction::QEQEM) {
    if (auto err = emitQEM(config, outputStream, std::move(payload), moduleOp,
                           targetCompilationManager, sourceBuffer, timing))
      return err;
  }

  return llvm::Error::success();
}

/// @brief Emit all given diagnostics, return true if there are errors
///
///        Uses qssc::emitDiagnostic to forward all diagnostics held by the
///        target to the python diagnostic callback. Also prints diagnostics to
///        llvm::errs to provide info in logs and for the binary. Returns true
///        iff the target contains error or fatal severity diagnostics.
/// @param diagnostics List of diagnostcs to emit
/// @param diagnosticCb Handle to python diagnostic callback
/// @param config Config data holding the verbosity level for output
/// @return True iff target contains any error or fatal diagnostics
bool emitDiagnosticsAndCheckForErrors(
    const qssc::DiagList &diagnostics,
    const qssc::OptDiagnosticCallback &diagnosticCb, const QSSConfig &config) {
  // NOLINTNEXTLINE(misc-const-correctness)
  bool foundError = false;
  for (auto &diag : diagnostics) {
    auto severity = diag.severity;
    switch (config.getVerbosityLevel()) {
    case QSSVerbosity::Error:
      switch (severity) {
      case Severity::Fatal:
      case Severity::Error:
        foundError = true;
        (void)qssc::emitDiagnostic(diagnosticCb, diag);
        llvm::errs() << diag.toString() << "\n";
        break;
      case Severity::Warning:
      case Severity::Info:
        break;
      default:
        llvm_unreachable("Unknown diagnostic severity");
      }
      break;
    case QSSVerbosity::Warn:
      switch (severity) {
      case Severity::Fatal:
      case Severity::Error:
        foundError = true;
        [[fallthrough]];
      case Severity::Warning:
        (void)qssc::emitDiagnostic(diagnosticCb, diag);
        llvm::errs() << diag.toString() << "\n";
        break;
      case Severity::Info:
        break;
      default:
        llvm_unreachable("Unknown diagnostic severity");
      }
      break;
    case QSSVerbosity::Info:
    case QSSVerbosity::Debug:
      switch (severity) {
      case Severity::Fatal:
      case Severity::Error:
        foundError = true;
        [[fallthrough]];
      case Severity::Warning:
      case Severity::Info:
        (void)qssc::emitDiagnostic(diagnosticCb, diag);
        llvm::errs() << diag.toString() << "\n";
        break;
      default:
        llvm_unreachable("Unknown diagnostic severity");
      }
      break;
    default:
      llvm_unreachable("Unknown verbosity level");
    } // end switch for config verbosity
  }   // end for diag in diagnostics list
  return foundError;
}

llvm::Error performCompileActions(llvm::raw_ostream &outputStream,
                                  std::unique_ptr<llvm::MemoryBuffer> buffer,
                                  DialectRegistry &registry,
                                  mlir::MLIRContext &context,
                                  const qssc::config::QSSConfig &config,
                                  mlir::TimingScope &timing,
                                  qssc::OptDiagnosticCallback diagnosticCb) {

  // Populate the context
  context.appendDialectRegistry(registry);
  context.allowUnregisteredDialects(config.shouldAllowUnregisteredDialects());
  context.printOpOnDiagnostic(!config.shouldVerifyDiagnostics());

  // Register LLVM dialect and all infrastructure required for translation to
  // LLVM IR
  mlir::registerBuiltinDialectTranslation(context);
  mlir::registerLLVMDialectTranslation(context);
  // Build the target for compilation
  auto targetResult = buildTarget(&context, config, timing);
  if (auto err = targetResult.takeError())
    return err;
  auto &target = targetResult.get();

  // Set up the input, which is loaded from a file by name or stdin
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  auto sourceBufferID =
      sourceMgr->AddNewSourceBuffer(std::move(buffer), llvm::SMLoc());

  const llvm::MemoryBuffer *sourceBuffer =
      sourceMgr->getMemoryBuffer(sourceBufferID);

  context.getDiagEngine().registerHandler([&](mlir::Diagnostic &diagnostic) {
    diagEngineHandler(diagnostic, diagnosticCb);
  });

  std::unique_ptr<qssc::payload::Payload> payload = nullptr;

  if (config.getEmitAction() == EmitAction::QEQEM &&
      !config.getTargetName().has_value())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "Unsupported target-specific payload: no target");
  if (config.getEmitAction() == EmitAction::QEM ||
      config.getEmitAction() == EmitAction::QEQEM) {
    const auto payloadType = (config.getEmitAction() == EmitAction::QEM)
                                 ? "ZIP"
                                 : config.getTargetName().value();
    auto payloadInfo =
        qssc::payload::registry::PayloadRegistry::lookupPluginInfo(payloadType);
    if (payloadInfo == std::nullopt)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Unsupported target-specific payload: " +
                                         payloadType);

    auto payloadName = config.getPayloadName().str();
    std::optional<qssc::payload::PayloadConfig> payloadConfig = std::nullopt;
    if (payloadName != "-")
      payloadConfig = {config.getPayloadName().str(),
                       config.getPayloadName().str(),
                       config.getVerbosityLevel()};

    payload = std::move(
        payloadInfo.value()->createPluginInstance(payloadConfig).get());
  }

  mlir::ModuleOp moduleOp;
  mlir::FallbackAsmResourceMap fallbackResourceMap;

  if (config.getInputType() == InputType::QASM) {

    mlir::TimingScope loadQASM3Timing = timing.nest("load-qasm3");
    if (config.getEmitAction() >= EmitAction::MLIR) {
      auto bufferIdentifier = sourceBuffer->getBufferIdentifier();

      LocationAttr sourceLoc;
      if ((bufferIdentifier == "" || bufferIdentifier == "<stdin>"))
        sourceLoc = UnknownLoc::get(&context);
      else
        sourceLoc = mlir::FileLineColLoc::get(&context, bufferIdentifier, 0, 0);

      moduleOp = mlir::ModuleOp::create(sourceLoc);
    }
    if (auto frontendError = qssc::frontend::openqasm3::parse(
            *sourceMgr, config.getEmitAction() == EmitAction::AST,
            config.getEmitAction() == EmitAction::ASTPretty,
            config.getEmitAction() >= EmitAction::MLIR, moduleOp, diagnosticCb,
            loadQASM3Timing))
      return frontendError;
    if (config.getEmitAction() < EmitAction::MLIR)
      return llvm::Error::success();
  } // if input == QASM

  // Parse mlir::parseSourceFile automatically differentiates between MLIR and
  // MLIR bytecode and treats them as interchangeable from the perspective of
  // source parsing.
  if (config.getInputType() == InputType::MLIR ||
      config.getInputType() == InputType::Bytecode) {

    mlir::TimingScope mlirParserTiming = timing.nest("parse-mlir");
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
                                     "Problem parsing source file to MLIR");

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

  if (auto err =
          applyEmitAction(config, outputStream, std::move(payload), context,
                          moduleOp, sourceBuffer, fallbackResourceMap,
                          targetCompilationManager, errorHandler, timing)) {
    emitDiagnosticsAndCheckForErrors(
        targetCompilationManager.takeTargetDiagnostics(), diagnosticCb, config);
    return err;
  }

  if (emitDiagnosticsAndCheckForErrors(
          targetCompilationManager.takeTargetDiagnostics(), diagnosticCb,
          config)) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Problems generating compiler output!");
  }

  return llvm::Error::success();
}

} // anonymous namespace

// The following implementation is based on that of MLIROptMain in the core
// MLIR project with the aim of standardizing CLI tooling and making forwards
// compatiability more straightforward

void qssc::registerAndParseCLIOptions(int argc, const char **argv,
                                      llvm::StringRef toolName,
                                      mlir::DialectRegistry &registry) {
  // Register all extensions
  mlir::registerAllExtensions(registry);

  registerCLOpts();
  // Register CL config builder prior to parsing
  CLIConfigBuilder::registerCLOptions(registry);
  llvm::cl::SetVersionPrinter(&printVersion);
  llvm::cl::ParseCommandLineOptions(argc, argv, toolName);
}

std::pair<std::string, std::string>
qssc::registerAndParseCLIToolOptions(int argc, const char **argv,
                                     llvm::StringRef toolName,
                                     mlir::DialectRegistry &registry) {

  static llvm::cl::opt<std::string> inputFilename(
      llvm::cl::Positional, llvm::cl::desc("<input file>"),
      llvm::cl::init("-"));

  static llvm::cl::opt<std::string> outputFilename(
      "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
      llvm::cl::init("-"));

  registerAndParseCLIOptions(argc, argv, toolName, registry);

  return std::make_pair(inputFilename.getValue(), outputFilename.getValue());
}

llvm::Error qssc::compileMain(llvm::raw_ostream &outputStream,
                              std::unique_ptr<llvm::MemoryBuffer> buffer,
                              DialectRegistry &registry,
                              const qssc::config::QSSConfig &config,
                              OptDiagnosticCallback diagnosticCb,
                              mlir::TimingScope &timing) {

  // The MLIR context for this compilation event.
  // Instantiate after parsing command line options.
  MLIRContext context{};

  std::unique_ptr<llvm::ThreadPool> threadPool;
  // Override default threadpool threads
  if (context.isMultithreadingEnabled() && config.getMaxThreads().has_value()) {
    llvm::ThreadPoolStrategy strategy;
    // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
    strategy.ThreadsRequested = config.getMaxThreads().value();
    threadPool = std::make_unique<llvm::ThreadPool>(strategy);
    context.setThreadPool(*threadPool.get());
  }

  qssc::config::setContextConfig(&context, config);

  return performCompileActions(outputStream, std::move(buffer), registry,
                               context, config, timing,
                               std::move(diagnosticCb));
}

llvm::Error qssc::compileMain(int argc, const char **argv,
                              llvm::StringRef inputFilename,
                              llvm::StringRef outputFilename,
                              mlir::DialectRegistry &registry,
                              OptDiagnosticCallback diagnosticCb) {

  llvm::InitLLVM const y(argc, argv);

  mlir::DefaultTimingManager tm;
  mlir::applyDefaultTimingManagerCLOptions(tm);
  // NOLINTNEXTLINE(misc-const-correctness)
  mlir::TimingScope timing = tm.getRootScope();

  mlir::TimingScope buildConfigTiming = timing.nest("build-config");
  auto configResult =
      qssc::config::buildToolConfig(inputFilename, outputFilename);
  if (auto err = configResult.takeError())
    return err;
  qssc::config::QSSConfig const config = configResult.get();
  buildConfigTiming.stop();

  if (config.shouldShowDialects()) {
    showDialects(registry);
    return llvm::Error::success();
  }

  if (config.shouldShowTargets()) {
    showTargets();
    return llvm::Error::success();
  }

  if (config.shouldShowPayloads()) {
    showPayloads();
    return llvm::Error::success();
  }

  if (config.shouldShowConfig()) {
    config.emit(llvm::outs());
    return llvm::Error::success();
  }

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
  if (!file)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to open input file: " +
                                       errorMessage);

  auto output = mlir::openOutputFile(outputFilename, &errorMessage);
  if (!output)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to open output file: " +
                                       errorMessage);

  if (auto err = compileMain(output->os(), std::move(file), registry, config,
                             std::move(diagnosticCb), timing))
    return err;

  // Keep the output file if the invocation of MlirOptMain was successful.
  output->keep();
  return llvm::Error::success();
}

llvm::Error qssc::compileMain(int argc, const char **argv,
                              llvm::StringRef toolName,
                              mlir::DialectRegistry &registry,
                              OptDiagnosticCallback diagnosticCb) {

  // Register and parse command line options.
  std::string inputFilename, outputFilename;
  std::tie(inputFilename, outputFilename) =
      registerAndParseCLIToolOptions(argc, argv, toolName, registry);

  return compileMain(argc, argv, inputFilename, outputFilename, registry,
                     std::move(diagnosticCb));
}

llvm::Error qssc::compileMain(int argc, const char **argv,
                              llvm::StringRef toolName,
                              OptDiagnosticCallback diagnosticCb) {
  // Register the standard passes with MLIR.
  // Must precede the command line parsing.
  if (auto err = qssc::dialect::registerPasses())
    return err;

  mlir::DialectRegistry registry;

  // Add the following to include *all* QSS core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  qssc::dialect::registerDialects(registry);

  return compileMain(argc, argv, toolName, registry, std::move(diagnosticCb));
}

//===------------------------ Parameter binding -------------------------===//

namespace {
class MapAngleArgumentSource : public qssc::arguments::ArgumentSource {

public:
  MapAngleArgumentSource(
      const std::unordered_map<std::string, double> &parameterMap)
      : parameterMap(parameterMap) {}

  qssc::arguments::ArgumentType
  getArgumentValue(llvm::StringRef name) const override {
    std::string const name_{name};
    auto pos = parameterMap.find(name_);

    if (pos == parameterMap.end())
      return std::nullopt;
    return pos->second;
  }

private:
  const std::unordered_map<std::string, double> &parameterMap;
};

llvm::Error
bindArguments_(std::string_view target, qssc::config::EmitAction action,
               std::string_view configPath, std::string_view moduleInput,
               std::string_view payloadOutputPath,
               std::unordered_map<std::string, double> const &arguments,
               bool treatWarningsAsErrors, bool enableInMemoryInput,
               std::string *inMemoryOutput,
               const qssc::OptDiagnosticCallback &onDiagnostic) {

  MLIRContext context{};

  qssc::hal::registry::TargetSystemInfo &targetInfo =
      *qssc::hal::registry::TargetSystemRegistry::lookupPluginInfo(target)
           .value_or(qssc::hal::registry::TargetSystemRegistry::
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

  MapAngleArgumentSource const source(arguments);

  auto factory =
      targetInst.get()->getBindArgumentsImplementationFactory(action);
  if ((!factory.has_value()) || (factory.value() == nullptr)) {
    return qssc::emitDiagnostic(
        onDiagnostic, qssc::Severity::Error,
        qssc::ErrorCategory::QSSLinkerNotImplemented,
        "Unable to load bind arguments implementation for target.");
  }
  qssc::arguments::BindArgumentsImplementationFactory &factoryRef =
      *factory.value();
  return qssc::arguments::bindArguments(
      moduleInput, payloadOutputPath, source, treatWarningsAsErrors,
      enableInMemoryInput, inMemoryOutput, factoryRef, onDiagnostic);
}

} // anonymous namespace

int qssc::bindArguments(
    std::string_view target, qssc::config::EmitAction action,
    std::string_view configPath, std::string_view moduleInput,
    std::string_view payloadOutputPath,
    std::unordered_map<std::string, double> const &arguments,
    bool treatWarningsAsErrors, bool enableInMemoryInput,
    std::string *inMemoryOutput,
    const qssc::OptDiagnosticCallback &onDiagnostic) {

  if (auto err =
          bindArguments_(target, action, configPath, moduleInput,
                         payloadOutputPath, arguments, treatWarningsAsErrors,
                         enableInMemoryInput, inMemoryOutput, onDiagnostic)) {
    llvm::logAllUnhandledErrors(std::move(err), llvm::errs());
    return 1;
  }
  return 0;
}
