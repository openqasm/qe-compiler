//===- QSSConfig.cpp - Config info --------------------*- C++ -*-----------===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file implements standard handling of QSSConfiguration options.
///
//===----------------------------------------------------------------------===//

#include "Config/QSSConfig.h"

#include "Config/CLIConfig.h"
#include "Config/EnvVarConfig.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/Plugins/DialectPlugin.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <ostream>
#include <string>
#include <utility>

using namespace qssc::config;

/// Verbosity levels for logging output
/// Error - Only error messages will be logged
/// Warn  - Warnings and errors *default
/// Info  - General information on compilation progress
/// Debug - Detailed output
const std::array<const std::string, _VerbosityCnt> verbosityToStr = {
    "Error", "Warn", "Info", "Debug"};

// For now emit in a pseudo-TOML format.
void qssc::config::QSSConfig::emit(llvm::raw_ostream &os) const {
  // Compiler configuration
  os << "[compiler]\n";
  os << "inputSource: " << getInputSource() << "\n";
  os << "directInput: " << isDirectInput() << "\n";
  os << "outputFilePath: " << getOutputFilePath() << "\n";
  os << "inputType: " << to_string(getInputType()) << "\n";
  os << "emitAction: " << to_string(getEmitAction()) << "\n";
  os << "targetName: "
     // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
     << (getTargetName().has_value() ? getTargetName().value() : "None")
     << "\n";
  os << "targetConfigPath: "
     // NOLINTNEXTLINE(bugprone-unchecked-optional-access)
     << (getTargetConfigPath().has_value() ? getTargetConfigPath().value()
                                           : "None")
     << "\n";
  os << "verbosity: "
     << verbosityToStr[static_cast<unsigned int>(getVerbosityLevel())] << "\n";
  os << "addTargetPasses: " << shouldAddTargetPasses() << "\n";
  os << "showTargets: " << shouldShowTargets() << "\n";
  os << "showPayloads: " << shouldShowPayloads() << "\n";
  os << "showConfig: " << shouldShowConfig() << "\n";
  os << "emitPlaintextPayload: " << shouldEmitPlaintextPayload() << "\n";
  os << "includeSource: " << shouldIncludeSource() << "\n";
  os << "compileTargetIR: " << shouldCompileTargetIR() << "\n";
  os << "bypassPayloadTargetCompilation: "
     << shouldBypassPayloadTargetCompilation() << "\n";
  os << "\n";

  // Mlir opt configuration
  os << "[opt]\n";
  os << "allowUnregisteredDialects: " << shouldAllowUnregisteredDialects()
     << "\n";
  os << "dumpPassPipeline: " << shouldDumpPassPipeline() << "\n";
  os << "emitBytecode: " << shouldEmitBytecode() << "\n";
  os << "bytecodeEmitVersion: " << bytecodeVersionToEmit() << "\n";
  os << "irdlFile: " << getIrdlFile() << "\n";
  os << "runReproducer: " << shouldRunReproducer() << "\n";
  os << "showDialects: " << shouldShowDialects() << "\n";
  os << "splitInputFile: " << shouldSplitInputFile() << "\n";
  os << "useExplicitModule: " << shouldUseExplicitModule() << "\n";
  os << "verifyDiagnostics: " << shouldVerifyDiagnostics() << "\n";
  os << "verifyPasses: " << shouldVerifyPasses() << "\n";
  os << "verifyRoundTrip: " << shouldVerifyRoundtrip() << "\n";
  os << "\n";
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const qssc::config::QSSConfig &config) {
  config.emit(os);
  return os;
}

std::ostream &operator<<(std::ostream &os,
                         const qssc::config::QSSConfig &config) {
  llvm::raw_os_ostream raw_os(os);
  config.emit(raw_os);
  return os;
}

namespace {
/// Mapping of registered MLIRContext configurations.
/// QUESTION: Rather than a global registry it seems like it would be much
/// better to inherit the MLIRContext as QSSContext and set the configuration on
/// this? Alternatively the QSSContext could own the MLIRContext?
static llvm::ManagedStatic<llvm::DenseMap<mlir::MLIRContext *, QSSConfig>>
    contextConfigs{};

} // anonymous namespace

void qssc::config::setContextConfig(mlir::MLIRContext *context,
                                    const QSSConfig &config) {
  (*contextConfigs)[context] = config;
}

llvm::Expected<const QSSConfig &>
qssc::config::getContextConfig(mlir::MLIRContext *context) {
  auto it = contextConfigs->find(context);
  if (it != contextConfigs->end())
    return it->getSecond();

  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      "Error: no config registered for the given context.\n");
}

llvm::Expected<QSSConfig> QSSConfigBuilder::buildConfig() {
  QSSConfig config;
  if (auto e = populateConfig(config))
    // Explicit move required for some systems as automatic move
    // is not recognized.
    return std::move(e);
  return config;
}

std::string qssc::config::to_string(const EmitAction &inAction) {
  switch (inAction) {
  case qssc::config::EmitAction::AST:
    return "ast";
    break;
  case EmitAction::ASTPretty:
    return "ast-pretty";
    break;
  case EmitAction::MLIR:
    return "mlir";
    break;
  case EmitAction::WaveMem:
    return "wmem";
    break;
  case EmitAction::QEM:
    return "qem";
    break;
  case EmitAction::QEQEM:
    return "qeqem";
    break;
  default:
    return "none";
    break;
  }
  return "none";
}

std::string qssc::config::to_string(const FileExtension &inExt) {
  switch (inExt) {
  case FileExtension::AST:
    return "ast";
    break;
  case FileExtension::ASTPretty:
    return "ast-pretty";
    break;
  case FileExtension::QASM:
    return "qasm";
    break;
  case FileExtension::MLIR:
    return "mlir";
    break;
  case FileExtension::WaveMem:
    return "wmem";
    break;
  case FileExtension::QEM:
    return "qem";
    break;
  case FileExtension::QEQEM:
    return "qeqem";
    break;
  default:
    return "none";
    break;
  }
  return "none";
}

std::string qssc::config::to_string(const InputType &inputType) {
  switch (inputType) {
  case InputType::QASM:
    return "qasm";
    break;
  case InputType::MLIR:
    return "mlir";
    break;
  default:
    return "none";
    break;
  }
  return "none";
}

InputType qssc::config::fileExtensionToInputType(const FileExtension &inExt) {
  switch (inExt) {
  case FileExtension::QASM:
    return InputType::QASM;
    break;
  case FileExtension::MLIR:
    return InputType::MLIR;
    break;
  default:
    break;
  }
  return InputType::None;
}

EmitAction qssc::config::fileExtensionToAction(const FileExtension &inExt) {
  switch (inExt) {
  case FileExtension::AST:
    return EmitAction::AST;
    break;
  case FileExtension::ASTPretty:
    return EmitAction::ASTPretty;
    break;
  case FileExtension::MLIR:
    return EmitAction::MLIR;
    break;
  case FileExtension::WaveMem:
    return EmitAction::WaveMem;
    break;
  case FileExtension::QEM:
    return EmitAction::QEM;
    break;
  case FileExtension::QEQEM:
    return EmitAction::QEQEM;
    break;
  default:
    break;
  }
  return EmitAction::None;
}

FileExtension qssc::config::strToFileExtension(const llvm::StringRef extStr) {
  if (extStr == "ast" || extStr == "AST")
    return FileExtension::AST;
  if (extStr == "ast-pretty" || extStr == "AST-PRETTY")
    return FileExtension::ASTPretty;
  if (extStr == "qasm" || extStr == "QASM")
    return FileExtension::QASM;
  if (extStr == "mlir" || extStr == "MLIR")
    return FileExtension::MLIR;
  if (extStr == "wmem" || extStr == "WMEM")
    return FileExtension::WaveMem;
  if (extStr == "qem" || extStr == "QEM")
    return FileExtension::QEM;
  if (extStr == "qeqem" || extStr == "QEQEM")
    return FileExtension::QEQEM;
  return FileExtension::None;
}

FileExtension qssc::config::getExtension(const llvm::StringRef inStr) {
  auto pos = inStr.find_last_of('.');
  if (pos < inStr.size())
    return strToFileExtension(inStr.substr(pos + 1));
  return FileExtension::None;
}

mlir::LogicalResult
qssc::config::loadDialectPlugin(const std::string &pluginPath,
                                mlir::DialectRegistry &registry) {
  auto plugin = mlir::DialectPlugin::load(pluginPath);
  if (!plugin)
    return mlir::failure();
  ;
  plugin.get().registerDialectRegistryCallbacks(registry);
  return mlir::success();
}

mlir::LogicalResult
qssc::config::loadPassPlugin(const std::string &pluginPath) {
  auto plugin = mlir::PassPlugin::load(pluginPath);
  if (!plugin)
    return mlir::failure();
  plugin.get().registerPassRegistryCallbacks();
  return mlir::success();
}

llvm::Expected<qssc::config::QSSConfig> qssc::config::buildToolConfig() {
  // First populate the configuration from default values then
  // environment variables.
  auto config = EnvVarConfigBuilder().buildConfig();
  if (auto err = config.takeError())
    // Explicit move required for some systems as automatic move
    // is not recognized.
    return std::move(err);

  // Apply CLI options of top of the configuration constructed above.
  if (auto err = CLIConfigBuilder().populateConfig(*config))
    // Explicit move required for some systems as automatic move
    // is not recognized.
    return std::move(err);

  return config;
}
