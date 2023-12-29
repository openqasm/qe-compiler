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

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"

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

} // anonymous namespace

class BytecodeVersionParser : public cl::parser<std::optional<int64_t>> {
public:
  BytecodeVersionParser(cl::Option &O)
      : cl::parser<std::optional<int64_t>>(O) {}

  bool parse(cl::Option &O, StringRef /*argName*/, StringRef arg,
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
struct QSSConfigCLOptions : public QSSConfig {
  QSSConfigCLOptions() {

    // qss-compiler options
    static llvm::cl::opt<std::optional<std::string>, /*ExternalStorage=*/true> targetConfigPath_(
        "config",
        llvm::cl::desc("Path to configuration file or directory (depends on the "
                      "target), - means use the config service"),
        llvm::cl::value_desc("path"), cl::location(targetConfigPath), cl::init(std::nullopt), llvm::cl::cat(getQSSCCLCategory()));

    static llvm::cl::opt<std::optional<std::string>, /*ExternalStorage=*/true>
        targetName_("target",
                  llvm::cl::desc(
                      "Target architecture. Required for machine code generation."),
                  llvm::cl::value_desc("targetName"), cl::location(targetName), cl::init(std::nullopt),
                  llvm::cl::cat(getQSSCCLCategory()));

    static llvm::cl::opt<bool, /*ExternalStorage=*/true> addTargetPasses(
        "add-target-passes", llvm::cl::desc("Add target-specific passes"),
        cl::location(addTargetPassesFlag), llvm::cl::init(true), llvm::cl::cat(getQSSCCLCategory()));


    // mlir-opt options

    static cl::opt<bool, /*ExternalStorage=*/true> allowUnregisteredDialects(
        "allow-unregistered-dialect",
        cl::desc("Allow operation with no registered dialects"),
        cl::location(allowUnregisteredDialectsFlag), cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static cl::opt<bool, /*ExternalStorage=*/true> dumpPassPipeline(
        "dump-pass-pipeline", cl::desc("Print the pipeline that will be run"),
        cl::location(dumpPassPipelineFlag), cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static cl::opt<bool, /*ExternalStorage=*/true> emitBytecode(
        "emit-bytecode", cl::desc("Emit bytecode when generating output"),
        cl::location(emitBytecodeFlag), cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static cl::opt<std::optional<int64_t>, /*ExternalStorage=*/true,
                   BytecodeVersionParser>
        bytecodeVersion(
            "emit-bytecode-version",
            cl::desc("Use specified bytecode when generating output"),
            cl::location(emitBytecodeVersion), cl::init(std::nullopt), llvm::cl::cat(getQSSOptCLCategory()));

    static cl::opt<std::string, /*ExternalStorage=*/true> irdlFile(
        "irdl-file",
        cl::desc("IRDL file to register before processing the input"),
        cl::location(irdlFileFlag), cl::init(""), cl::value_desc("filename"), llvm::cl::cat(getQSSOptCLCategory()));

    static cl::opt<bool, /*ExternalStorage=*/true> enableDebuggerHook(
        "mlir-enable-debugger-hook",
        cl::desc("Enable Debugger hook for debugging MLIR Actions"),
        cl::location(enableDebuggerActionHookFlag), cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static cl::opt<bool, /*ExternalStorage=*/true> explicitModule(
        "no-implicit-module",
        cl::desc("Disable implicit addition of a top-level module op during "
                 "parsing"),
        cl::location(useExplicitModuleFlag), cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static cl::opt<bool, /*ExternalStorage=*/true> runReproducer(
        "run-reproducer", cl::desc("Run the pipeline stored in the reproducer"),
        cl::location(runReproducerFlag), cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static cl::opt<bool, /*ExternalStorage=*/true> showDialects(
        "show-dialects",
        cl::desc("Print the list of registered dialects and exit"),
        cl::location(showDialectsFlag), cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static cl::opt<bool, /*ExternalStorage=*/true> splitInputFile(
        "split-input-file",
        cl::desc("Split the input file into pieces and process each "
                 "chunk independently"),
        cl::location(splitInputFileFlag), cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static cl::opt<bool, /*ExternalStorage=*/true> verifyDiagnostics(
        "verify-diagnostics",
        cl::desc("Check that emitted diagnostics match "
                 "expected-* lines on the corresponding line"),
        cl::location(verifyDiagnosticsFlag), cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static cl::opt<bool, /*ExternalStorage=*/true> verifyPasses(
        "verify-each",
        cl::desc("Run the verifier after each transformation pass"),
        cl::location(verifyPassesFlag), cl::init(true), llvm::cl::cat(getQSSOptCLCategory()));

    static cl::opt<bool, /*ExternalStorage=*/true> verifyRoundtrip(
        "verify-roundtrip",
        cl::desc("Round-trip the IR after parsing and ensure it succeeds"),
        cl::location(verifyRoundtripFlag), cl::init(false), llvm::cl::cat(getQSSOptCLCategory()));

    static cl::list<std::string> passPlugins(
        "load-pass-plugin", cl::desc("Load passes from plugin library"), llvm::cl::cat(getQSSOptCLCategory()));
    /// Set the callback to load a pass plugin.
    passPlugins.setCallback([&](const std::string &pluginPath) {
      auto plugin = PassPlugin::load(pluginPath);
      if (!plugin) {
        errs() << "Failed to load passes from '" << pluginPath
               << "'. Request ignored.\n";
        return;
      }
      plugin.get().registerPassRegistryCallbacks();
    });

    static cl::list<std::string> dialectPlugins(
        "load-dialect-plugin", cl::desc("Load dialects from plugin library"), llvm::cl::cat(getQSSOptCLCategory()));
    this->dialectPlugins = std::addressof(dialectPlugins);

    static PassPipelineCLParser passPipeline("", "Compiler passes to run", "p");
    setPassPipelineParser(passPipeline);
  }

  /// Set the callback to load a dialect plugin.
  void setDialectPluginsCallback(DialectRegistry &registry);

  /// Pointer to static dialectPlugins variable in constructor, needed by
  /// setDialectPluginsCallback(DialectRegistry&).
  cl::list<std::string> *dialectPlugins = nullptr;
};


llvm::cl::OptionCategory &qssc::config::getQSSCCLCategory() { return qsscCat_; }
llvm::cl::OptionCategory &qssc::config::getQSSOptCLCategory() { return optCat_; }
