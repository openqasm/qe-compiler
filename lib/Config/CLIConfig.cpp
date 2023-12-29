//===- CLIConfigBuilder.cpp - QSSConfig from the CLI ------*- C++ -*-------===//
//
// (C) Copyright IBM 2023.
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

#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"

#include <string>

using namespace qssc::config;

namespace {
// The space below at the front of the string causes this category to be printed
// first
llvm::cl::OptionCategory qsscCat_(" qss-compiler options",
          "Options that control high-level behavior of QSS Compiler");

llvm::cl::OptionCategory optCat_(" qss-compiler options: opt",
            "Options that control behaviour inherited from mlir-opt.");


class BytecodeVersionParser : public llvm::cl::parser<std::optional<int64_t>> {
public:
  BytecodeVersionParser(llvm::cl::Option &O)
      : llvm::cl::parser<std::optional<int64_t>>(O) {}

  bool parse(llvm::cl::Option &O, llvm::StringRef /*argName*/, llvm::StringRef arg,
             std::optional<int64_t> &v) {
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
/// As the implementation is anonymous we recreate the population of the configuration here.
struct QSSConfigCLOptions : public QSSConfig {
  QSSConfigCLOptions() {

    // qss-compiler options
    static llvm::cl::opt<std::optional<std::string>, /*ExternalStorage=*/true> targetConfigPath_(
        "config",
        llvm::cl::desc("Path to configuration file or directory (depends on the "
                      "target), - means use the config service"),
        llvm::cl::value_desc("path"), llvm::cl::location(targetConfigPath), llvm::cl::init(std::nullopt), llvm::cl::cat(getQSSCCLCategory()));

    static llvm::cl::opt<std::optional<std::string>, /*ExternalStorage=*/true>
        targetName_("target",
                  llvm::cl::desc(
                      "Target architecture. Required for machine code generation."),
                  llvm::cl::value_desc("targetName"), llvm::cl::location(targetName), llvm::cl::init(std::nullopt),
                  llvm::cl::cat(getQSSCCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> addTargetPasses(
        "add-target-passes", llvm::cl::desc("Add target-specific passes"),
        llvm::cl::location(addTargetPassesFlag), llvm::cl::init(true), llvm::cl::cat(getQSSCCLCategory()));


    // mlir-opt options

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> allowUnregisteredDialects(
        "allow-unregistered-dialect",
        llvm::cl::desc("Allow operation with no registered dialects"),
        llvm::cl::location(allowUnregisteredDialectsFlag), llvm::cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> dumpPassPipeline(
        "dump-pass-pipeline", llvm::cl::desc("Print the pipeline that will be run"),
        llvm::cl::location(dumpPassPipelineFlag), llvm::cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> emitBytecode(
        "emit-bytecode", llvm::cl::desc("Emit bytecode when generating output"),
        llvm::cl::location(emitBytecodeFlag), llvm::cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<std::optional<int64_t>, /*ExternalStorage=*/true,
                   BytecodeVersionParser>
        bytecodeVersion(
            "emit-bytecode-version",
            llvm::cl::desc("Use specified bytecode when generating output"),
            llvm::cl::location(emitBytecodeVersion), llvm::cl::init(std::nullopt), llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<std::string, /*ExternalStorage=*/true> irdlFile(
        "irdl-file",
        llvm::cl::desc("IRDL file to register before processing the input"),
        llvm::cl::location(irdlFileFlag), llvm::cl::init(""), llvm::cl::value_desc("filename"), llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> enableDebuggerHook(
        "mlir-enable-debugger-hook",
        llvm::cl::desc("Enable Debugger hook for debugging MLIR Actions"),
        llvm::cl::location(enableDebuggerActionHookFlag), llvm::cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> explicitModule(
        "no-implicit-module",
        llvm::cl::desc("Disable implicit addition of a top-level module op during "
                 "parsing"),
        llvm::cl::location(useExplicitModuleFlag), llvm::cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> runReproducer(
        "run-reproducer", llvm::cl::desc("Run the pipeline stored in the reproducer"),
        llvm::cl::location(runReproducerFlag), llvm::cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> showDialects(
        "show-dialects",
        llvm::cl::desc("Print the list of registered dialects and exit"),
        llvm::cl::location(showDialectsFlag), llvm::cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> splitInputFile(
        "split-input-file",
        llvm::cl::desc("Split the input file into pieces and process each "
                 "chunk independently"),
        llvm::cl::location(splitInputFileFlag), llvm::cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> verifyDiagnostics(
        "verify-diagnostics",
        llvm::cl::desc("Check that emitted diagnostics match "
                 "expected-* lines on the corresponding line"),
        llvm::cl::location(verifyDiagnosticsFlag), llvm::cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> verifyPasses(
        "verify-each",
        llvm::cl::desc("Run the verifier after each transformation pass"),
        llvm::cl::location(verifyPassesFlag), llvm::cl::init(true), llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> verifyRoundtrip(
        "verify-roundtrip",
        llvm::cl::desc("Round-trip the IR after parsing and ensure it succeeds"),
        llvm::cl::location(verifyRoundtripFlag), llvm::cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static llvm::cl::list<std::string> passPlugins(
        "load-pass-plugin", llvm::cl::desc("Load passes from plugin library"), llvm::cl::cat(getQSSOptCLCategory()));
    /// Set the callback to load a pass plugin.
    passPlugins.setCallback([&](const std::string &pluginPath) {
      auto plugin = mlir::PassPlugin::load(pluginPath);
      if (!plugin) {
        llvm::errs() << "Failed to load passes from '" << pluginPath
               << "'. Request ignored.\n";
        return;
      }
      plugin.get().registerPassRegistryCallbacks();
    });

    static llvm::cl::list<std::string> dialectPlugins(
        "load-dialect-plugin", llvm::cl::desc("Load dialects from plugin library"), llvm::cl::cat(getQSSOptCLCategory()));
    this->dialectPlugins = std::addressof(dialectPlugins);

    static mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run", "p");
    setPassPipelineParser(passPipeline);
  }

  /// Set the callback to load a dialect plugin.
  void setDialectPluginsCallback(mlir::DialectRegistry &registry) {
  dialectPlugins->setCallback([&](const std::string &pluginPath) {
    auto plugin = mlir::DialectPlugin::load(pluginPath);
    if (!plugin) {
      llvm::errs() << "Failed to load dialect plugin from '" << pluginPath
             << "'. Request ignored.\n";
      return;
    };
    plugin.get().registerDialectRegistryCallbacks(registry);
  });
}

  /// Pointer to static dialectPlugins variable in constructor, needed by
  /// setDialectPluginsCallback(DialectRegistry&).
  llvm::cl::list<std::string> *dialectPlugins = nullptr;
};

} // anonymous namespace

llvm::ManagedStatic<QSSConfigCLOptions> clOptionsConfig;


llvm::cl::OptionCategory &qssc::config::getQSSCCLCategory() { return qsscCat_; }
llvm::cl::OptionCategory &qssc::config::getQSSOptCLCategory() { return optCat_; }


CLIConfigBuilder::CLIConfigBuilder(mlir::DialectRegistry &registry) {
  clOptionsConfig->setDialectPluginsCallback(registry);
  mlir::tracing::DebugConfig::registerCLOptions();
}

llvm::Expected<QSSConfig> CLIConfigBuilder::buildConfig() {
  clOptionsConfig->setDebugConfig(mlir::tracing::DebugConfig::createFromCLOptions());
  return *clOptionsConfig;
}

llvm::Error CLIConfigBuilder::populateConfig(QSSConfig &config) {
  config.splitInputFileFlag = clOptionsConfig->splitInputFileFlag;

  return llvm::Error::success();
}
