//===- CLIConfigBuilder.cpp - QSSConfig from the CLI ------*- C++ -*-------===//
//
// (C) Copyright IBM 2023, 2024.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file implements building the configuration from the CLI.
///
//===----------------------------------------------------------------------===//

#include "Config/CLIConfig.h"
#include "Config/QSSConfig.h"

#include "mlir/Debug/CLOptionsSetup.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

using namespace qssc::config;

namespace {
// The space below at the front of the string causes this category to be printed
// first
llvm::cl::OptionCategory
    qsscCat_(" qss-compiler options",
             "Options that control high-level behavior of QSS Compiler");

llvm::cl::OptionCategory
    optCat_(" qss-compiler options: opt",
            "Options that control behaviour inherited from mlir-opt.");

class BytecodeVersionParser : public llvm::cl::parser<std::optional<int64_t>> {
public:
  BytecodeVersionParser(llvm::cl::Option &O)
      : llvm::cl::parser<std::optional<int64_t>>(O) {}

  bool parse(llvm::cl::Option &O, llvm::StringRef /*argName*/,
             llvm::StringRef arg, std::optional<int64_t> &v) {
    long long w;
    if (getAsSignedInteger(arg, 10, w))
      return O.error("Invalid argument '" + arg +
                     "', only integer is supported.");
    v = w;
    return false;
  }
};

/// This class is intended to manage the handling of command line options for
/// creating a qss-compiler mlir-opt based config. This is a singleton.
/// The implementation closely follows that of
/// https://github.com/llvm/llvm-project/blob/llvmorg-17.0.6/mlir/lib/Tools/mlir-opt/MlirOptMain.cpp
/// As the implementation is anonymous we recreate the population of the
/// configuration here.
struct QSSConfigCLOptions : public QSSConfig {
  QSSConfigCLOptions() {

    // qss-compiler options
    static llvm::cl::opt<std::string, /*ExternalStorage=*/true> const
        inputSource_(llvm::cl::Positional,
                     llvm::cl::desc("Input filename or program source"),
                     llvm::cl::location(inputSource), llvm::cl::init("-"),
                     llvm::cl::cat(qssc::config::getQSSCCLCategory()));

    static llvm::cl::opt<std::string, /*ExternalStorage=*/true> const
        outputFilename("o", llvm::cl::desc("Output filename"),
                       llvm::cl::value_desc("filename"),
                       llvm::cl::location(outputFilePath), llvm::cl::init("-"),
                       llvm::cl::cat(qssc::config::getQSSCCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> const directInput(
        "direct",
        llvm::cl::desc("Accept the input program directly as a string"),
        llvm::cl::location(directInputFlag),
        llvm::cl::cat(qssc::config::getQSSCCLCategory()));

    static llvm::cl::opt<enum InputType, /*ExternalStorage=*/true> const
        inputType_("X", llvm::cl::location(inputType),
                   llvm::cl::desc("Specify the kind of input desired"),
                   llvm::cl::values(clEnumValN(
                       InputType::QASM, "qasm",
                       "load the input file as an OpenQASM 3.0 source")),
                   llvm::cl::values(
                       clEnumValN(InputType::MLIR, "mlir",
                                  "load the input file as an MLIR file")));

    static llvm::cl::opt<enum EmitAction, /*ExternalStorage=*/true> const
        emitAction_(
            "emit", llvm::cl::location(emitAction),
            llvm::cl::desc("Select the kind of output desired"),
            llvm::cl::values(
                clEnumValN(EmitAction::AST, "ast", "output the AST dump")),
            llvm::cl::values(clEnumValN(EmitAction::ASTPretty, "ast-pretty",
                                        "pretty print the AST")),
            llvm::cl::values(
                clEnumValN(EmitAction::MLIR, "mlir", "output the MLIR dump")),
            llvm::cl::values(clEnumValN(EmitAction::WaveMem, "wavemem",
                                        "output the waveform memory")),
            llvm::cl::values(
                clEnumValN(EmitAction::QEM, "qem",
                           "generate a quantum executable module (qem) "
                           "for execution on hardware")),
            llvm::cl::values(clEnumValN(
                EmitAction::QEQEM, "qe-qem",
                "generate a target-specific quantum executable module (qeqem) "
                "for execution on hardware")));

    static llvm::cl::opt<std::string> targetConfigPath_(
        "config",
        llvm::cl::desc(
            "Path to configuration file or directory (depends on the "
            "target), - means use the config service"),
        llvm::cl::value_desc("path"), llvm::cl::cat(getQSSCCLCategory()));

    targetConfigPath_.setCallback([&](const std::string &config) {
      if (config != "")
        targetConfigPath = config;
    });

    static llvm::cl::opt<std::string> targetName_(
        "target",
        llvm::cl::desc(
            "Target architecture. Required for machine code generation."),
        llvm::cl::value_desc("targetName"), llvm::cl::cat(getQSSCCLCategory()));

    targetName_.setCallback([&](const std::string &target) {
      if (target != "")
        targetName = target;
    });

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> const addTargetPasses(
        "add-target-passes", llvm::cl::desc("Add target-specific passes"),
        llvm::cl::location(addTargetPassesFlag), llvm::cl::init(true),
        llvm::cl::cat(getQSSCCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> const showTargets(
        "show-targets", llvm::cl::desc("Print the list of registered targets"),
        llvm::cl::location(showTargetsFlag), llvm::cl::init(false),
        llvm::cl::cat(qssc::config::getQSSCCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> const showPayloads(
        "show-payloads",
        llvm::cl::desc("Print the list of registered payloads"),
        llvm::cl::location(showPayloadsFlag), llvm::cl::init(false),
        llvm::cl::cat(qssc::config::getQSSCCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> const showConfig(
        "show-config",
        llvm::cl::desc("Print the loaded compiler configuration."),
        llvm::cl::location(showConfigFlag), llvm::cl::init(false),
        llvm::cl::cat(qssc::config::getQSSCCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> const plaintextPayload(
        "plaintext-payload", llvm::cl::desc("Write the payload in plaintext"),
        llvm::cl::location(emitPlaintextPayloadFlag), llvm::cl::init(false),
        llvm::cl::cat(qssc::config::getQSSCCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> const includeSource(
        "include-source",
        llvm::cl::desc("Write the input source into the payload"),
        llvm::cl::location(includeSourceFlag), llvm::cl::init(false),
        llvm::cl::cat(qssc::config::getQSSCCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> const compileTargetIr(
        "compile-target-ir",
        llvm::cl::desc("Apply the target's IR compilation"),
        llvm::cl::location(compileTargetIRFlag), llvm::cl::init(false),
        llvm::cl::cat(qssc::config::getQSSCCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> const
        bypassPayloadTargetCompilation(
            "bypass-payload-target-compilation",
            llvm::cl::desc(
                "Bypass target compilation during payload generation."),
            llvm::cl::location(bypassPayloadTargetCompilationFlag),
            llvm::cl::init(false),
            llvm::cl::cat(qssc::config::getQSSCCLCategory()));

    // mlir-opt options

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> const
        allowUnregisteredDialects(
            "allow-unregistered-dialect",
            llvm::cl::desc("Allow operation with no registered dialects"),
            llvm::cl::location(allowUnregisteredDialectsFlag),
            llvm::cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> const dumpPassPipeline(
        "dump-pass-pipeline",
        llvm::cl::desc("Print the pipeline that will be run"),
        llvm::cl::location(dumpPassPipelineFlag), llvm::cl::init(false),
        llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> const emitBytecode(
        "emit-bytecode", llvm::cl::desc("Emit bytecode when generating output"),
        llvm::cl::location(emitBytecodeFlag), llvm::cl::init(false),
        llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<std::optional<int64_t>, /*ExternalStorage=*/true,
                         BytecodeVersionParser> const
        bytecodeVersion(
            "emit-bytecode-version",
            llvm::cl::desc("Use specified bytecode when generating output"),
            llvm::cl::location(emitBytecodeVersion),
            llvm::cl::init(std::nullopt), llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<std::string, /*ExternalStorage=*/true> const irdlFile(
        "irdl-file",
        llvm::cl::desc("IRDL file to register before processing the input"),
        llvm::cl::location(irdlFileFlag), llvm::cl::init(""),
        llvm::cl::value_desc("filename"), llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> const
        enableDebuggerHook(
            "mlir-enable-debugger-hook",
            llvm::cl::desc("Enable Debugger hook for debugging MLIR Actions"),
            llvm::cl::location(enableDebuggerActionHookFlag),
            llvm::cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> const explicitModule(
        "no-implicit-module",
        llvm::cl::desc(
            "Disable implicit addition of a top-level module op during "
            "parsing"),
        llvm::cl::location(useExplicitModuleFlag), llvm::cl::init(false),
        llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> const runReproducer(
        "run-reproducer",
        llvm::cl::desc("Run the pipeline stored in the reproducer"),
        llvm::cl::location(runReproducerFlag), llvm::cl::init(false),
        llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> const showDialects(
        "show-dialects",
        llvm::cl::desc("Print the list of registered dialects and exit"),
        llvm::cl::location(showDialectsFlag), llvm::cl::init(false),
        llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> const splitInputFile(
        "split-input-file",
        llvm::cl::desc("Split the input file into pieces and process each "
                       "chunk independently"),
        llvm::cl::location(splitInputFileFlag), llvm::cl::init(false),
        llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> const
        verifyDiagnostics(
            "verify-diagnostics",
            llvm::cl::desc("Check that emitted diagnostics match "
                           "expected-* lines on the corresponding line"),
            llvm::cl::location(verifyDiagnosticsFlag), llvm::cl::init(false),
            llvm::cl::cat(getQSSOptCLCategory()));

#ifndef NOVERIFY
#define VERIFY_PASSES_DEFAULT true
#else
#define VERIFY_PASSES_DEFAULT false
#endif
    static llvm::cl::opt<bool, /*ExternalStorage=*/true> const verifyPasses(
        "verify-each",
        llvm::cl::desc("Run the verifier after each transformation pass"),
        llvm::cl::location(verifyPassesFlag),
        llvm::cl::init(VERIFY_PASSES_DEFAULT),
        llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> const verifyRoundtrip(
        "verify-roundtrip",
        llvm::cl::desc(
            "Round-trip the IR after parsing and ensure it succeeds"),
        llvm::cl::location(verifyRoundtripFlag), llvm::cl::init(false),
        llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::list<std::string> passPlugins_(
        "load-pass-plugin",
        llvm::cl::desc("Load passes from plugin library. It is required that "
                       "the pass be specified to be loaded before all usages "
                       "of dynamic CL arguments."),
        llvm::cl::cat(getQSSOptCLCategory()));
    /// Set the callback to load a pass plugin.
    passPlugins_.setCallback([&](const std::string &pluginPath) {
      passPlugins.push_back(pluginPath);
      if (mlir::failed(loadPassPlugin(pluginPath)))
        llvm::errs() << "Failed to load passes from '" << pluginPath
                     << "'. Request ignored.\n";
    });

    static llvm::cl::list<std::string> dialectPlugins_(
        "load-dialect-plugin",
        llvm::cl::desc("Load dialects from plugin library. It is required that "
                       "the dialect be specified to be loaded before all "
                       "usages of dynamic CL arguments"),
        llvm::cl::cat(getQSSOptCLCategory()));
    this->dialectPlugins_ = std::addressof(dialectPlugins_);

    static mlir::PassPipelineCLParser const passPipeline(
        "", "Compiler passes to run", "p");
    setPassPipelineParser(passPipeline);

    static llvm::cl::opt<enum QSSVerbosity, /*ExternalStorage=*/true> const
        verbosity(
            "verbosity", llvm::cl::location(verbosityLevel),
            llvm::cl::init(QSSVerbosity::_VerbosityCnt),
            llvm::cl::desc("Set verbosity level for output, default is warn"),
            llvm::cl::values(
                clEnumValN(QSSVerbosity::Error, "error", "Emit only errors")),
            llvm::cl::values(
                clEnumValN(QSSVerbosity::Warn, "warn", "Also emit warnings")),
            llvm::cl::values(clEnumValN(QSSVerbosity::Info, "info",
                                        "Also emit informational messages")),
            llvm::cl::values(clEnumValN(QSSVerbosity::Debug, "debug",
                                        "Also emit debug messages")),
            llvm::cl::cat(qssc::config::getQSSOptCLCategory()));
  }

  /// Pointer to static dialectPlugins variable in constructor, needed by
  /// setDialectPluginsCallback(DialectRegistry&).
  llvm::cl::list<std::string> *dialectPlugins_ = nullptr;

  void setDialectPluginsCallback(mlir::DialectRegistry &registry) {
    dialectPlugins_->setCallback([&](const std::string &pluginPath) {
      dialectPlugins.push_back(pluginPath);
      if (mlir::failed(loadDialectPlugin(pluginPath, registry)))
        llvm::errs() << "Failed to load dialect from '" << pluginPath
                     << "'. Request ignored.\n";
    });
  }

  llvm::Error computeInputType() {
    if (getInputType() == InputType::None) {

      if (isDirectInput())
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "The input source format must be "
                                       "specified with -X for direct input.");

      setInputType(fileExtensionToInputType(getExtension(getInputSource())));
      if (getInputSource() != "-" && getInputType() == InputType::None) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "Unable to autodetect file extension type! Please specify the "
            "input type with -X");
      }
    }

    return llvm::Error::success();
  }

  llvm::Error computeOutputType() {
    if (getOutputFilePath() != "-") {
      EmitAction const extensionAction =
          fileExtensionToAction(getExtension(getOutputFilePath()));
      if (extensionAction == EmitAction::None &&
          emitAction == EmitAction::None) {
        llvm::errs() << "Cannot determine the file extension of the specified "
                        "output file "
                     << getOutputFilePath() << " defaulting to dumping MLIR\n";
        setEmitAction(EmitAction::MLIR);
      } else if (emitAction == EmitAction::None) {
        setEmitAction(extensionAction);
      } else if (extensionAction != getEmitAction()) {
        llvm::errs()
            << "Warning! The output type in the file extension doesn't "
               "match the output type specified by --emit!";
      }
    } else {
      if (emitAction == EmitAction::None)
        setEmitAction(EmitAction::MLIR);
    }

    return llvm::Error::success();
  }
};

} // anonymous namespace

llvm::ManagedStatic<QSSConfigCLOptions> clOptionsConfig;

llvm::cl::OptionCategory &qssc::config::getQSSCCLCategory() { return qsscCat_; }
llvm::cl::OptionCategory &qssc::config::getQSSOptCLCategory() {
  return optCat_;
}

CLIConfigBuilder::CLIConfigBuilder() {
  clOptionsConfig->setDebugConfig(
      mlir::tracing::DebugConfig::createFromCLOptions());
}

void CLIConfigBuilder::registerCLOptions(mlir::DialectRegistry &registry) {
  clOptionsConfig->setDialectPluginsCallback(registry);
  mlir::tracing::DebugConfig::registerCLOptions();
}

llvm::Error CLIConfigBuilder::populateConfig(QSSConfig &config) {

  config.setDebugConfig(clOptionsConfig->getDebugConfig());

  config.setPassPipelineSetupFn(clOptionsConfig->passPipelineCallback);

  if (auto err = clOptionsConfig->computeInputType())
    return err;

  if (auto err = clOptionsConfig->computeOutputType())
    return err;

  if (clOptionsConfig->verbosityLevel != QSSVerbosity::_VerbosityCnt)
    config.verbosityLevel = clOptionsConfig->verbosityLevel;

  // qss
  config.inputSource = clOptionsConfig->inputSource;
  config.directInputFlag = clOptionsConfig->directInputFlag;
  config.outputFilePath = clOptionsConfig->outputFilePath;
  config.inputType = clOptionsConfig->inputType;
  config.emitAction = clOptionsConfig->emitAction;
  if (clOptionsConfig->targetName.has_value())
    config.targetName = clOptionsConfig->targetName;
  if (clOptionsConfig->targetConfigPath.has_value())
    config.targetConfigPath = clOptionsConfig->targetConfigPath;

  config.addTargetPassesFlag = clOptionsConfig->addTargetPassesFlag;
  config.showTargetsFlag = clOptionsConfig->showTargetsFlag;
  config.showPayloadsFlag = clOptionsConfig->showPayloadsFlag;
  config.showConfigFlag = clOptionsConfig->showConfigFlag;
  config.emitPlaintextPayloadFlag = clOptionsConfig->emitPlaintextPayloadFlag;
  config.includeSourceFlag = clOptionsConfig->includeSourceFlag;
  config.compileTargetIRFlag = clOptionsConfig->compileTargetIRFlag;
  config.bypassPayloadTargetCompilationFlag =
      clOptionsConfig->bypassPayloadTargetCompilationFlag;
  config.passPlugins.insert(config.passPlugins.end(),
                            clOptionsConfig->passPlugins.begin(),
                            clOptionsConfig->passPlugins.end());
  config.dialectPlugins.insert(config.dialectPlugins.end(),
                               clOptionsConfig->dialectPlugins.begin(),
                               clOptionsConfig->dialectPlugins.end());

  // opt
  config.allowUnregisteredDialectsFlag =
      clOptionsConfig->allowUnregisteredDialectsFlag;
  config.dumpPassPipelineFlag = clOptionsConfig->dumpPassPipelineFlag;
  config.emitBytecodeFlag = clOptionsConfig->emitBytecodeFlag;
  config.irdlFileFlag = clOptionsConfig->irdlFileFlag;
  config.enableDebuggerActionHookFlag =
      clOptionsConfig->enableDebuggerActionHookFlag;
  config.useExplicitModuleFlag = clOptionsConfig->useExplicitModuleFlag;
  config.runReproducerFlag = clOptionsConfig->runReproducerFlag;
  config.showDialectsFlag = clOptionsConfig->showDialectsFlag;
  config.verifyDiagnosticsFlag = clOptionsConfig->verifyDiagnosticsFlag;
  config.verifyPassesFlag = clOptionsConfig->verifyPassesFlag;
  config.verifyRoundtripFlag = clOptionsConfig->verifyRoundtripFlag;
  config.splitInputFileFlag = clOptionsConfig->splitInputFileFlag;
  return llvm::Error::success();
}
