//===- QSSConfig.h - Global QSS config ----------------*- C++ -*-----------===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  A centralized API for configuration handling within the QSS infrastructure.
///
//===----------------------------------------------------------------------===//
#ifndef QSSC_QSSCONFIG_H
#define QSSC_QSSCONFIG_H

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include <iostream>
#include <optional>
#include <string>
#include <utility>

namespace qssc::config {

enum class EmitAction { None, AST, ASTPretty, MLIR, WaveMem, QEM, QEQEM };

enum class FileExtension {
  None,
  AST,
  ASTPretty,
  QASM,
  MLIR,
  WaveMem,
  QEM,
  QEQEM
};

enum class InputType { None, QASM, MLIR };

std::string to_string(const EmitAction &inExt);

std::string to_string(const FileExtension &inExt);

std::string to_string(const InputType &inType);

InputType fileExtensionToInputType(const FileExtension &inExt);

EmitAction fileExtensionToAction(const FileExtension &inExt);

FileExtension strToFileExtension(const llvm::StringRef extStr);

FileExtension getExtension(const llvm::StringRef inStr);

/// @brief The QSS configuration data structure that is to be used for global
/// configuration of the QSS infrastructure. This is to be used for static
/// options that are rarely changed for a system and do not need to be
/// dynamically extensible (such as pluggable TargetInstrument and their
/// configuration). This configuration is constructed from several sources such
/// as CLI, environment variables and possible configuration file formats
/// through QSSConfigBuilder implementations which apply successive views over
/// the configuration to produce the final configuration.
struct QSSConfig : mlir::MlirOptMainConfig {

public:
  friend class CLIConfigBuilder;
  friend class EnvVarConfigBuilder;

  QSSConfig &setInputSource(std::string source) {
    inputSource = std::move(source);
    return *this;
  }
  llvm::StringRef getInputSource() const { return inputSource; }

  QSSConfig &directInput(bool flag) {
    directInputFlag = flag;
    return *this;
  }
  bool isDirectInput() const { return directInputFlag; }

  QSSConfig &setOutputFilePath(std::string path) {
    outputFilePath = std::move(path);
    return *this;
  }
  llvm::StringRef getOutputFilePath() const { return outputFilePath; }

  QSSConfig &setTargetName(std::string name) {
    targetName = std::move(name);
    return *this;
  }
  std::optional<llvm::StringRef> getTargetName() const {
    if (targetName.has_value())
      return targetName.value();
    return std::nullopt;
  }

  QSSConfig &setTargetConfigPath(std::string path) {
    targetConfigPath = std::move(path);
    return *this;
  }
  std::optional<llvm::StringRef> getTargetConfigPath() const {
    if (targetConfigPath.has_value())
      return targetConfigPath.value();
    return std::nullopt;
  }

  QSSConfig &setInputType(InputType type) {
    inputType = type;
    return *this;
  }
  InputType getInputType() const { return inputType; }

  QSSConfig &setEmitAction(EmitAction action) {
    emitAction = action;
    return *this;
  }
  EmitAction getEmitAction() const { return emitAction; }

  QSSConfig &addTargetPasses(bool flag) {
    addTargetPassesFlag = flag;
    return *this;
  }
  bool shouldAddTargetPasses() const { return addTargetPassesFlag; }

  QSSConfig &showTargets(bool flag) {
    showTargetsFlag = flag;
    return *this;
  }
  bool shouldShowTargets() const { return showTargetsFlag; }

  QSSConfig &showPayloads(bool flag) {
    showPayloadsFlag = flag;
    return *this;
  }
  bool shouldShowPayloads() const { return showPayloadsFlag; }

  QSSConfig &showConfig(bool flag) {
    showConfigFlag = flag;
    return *this;
  }
  bool shouldShowConfig() const { return showConfigFlag; }

  QSSConfig &emitPlaintextPayload(bool flag) {
    emitPlaintextPayloadFlag = flag;
    return *this;
  }
  bool shouldEmitPlaintextPayload() const { return emitPlaintextPayloadFlag; }

  QSSConfig &includeSource(bool flag) {
    includeSourceFlag = flag;
    return *this;
  }
  bool shouldIncludeSource() const { return includeSourceFlag; }

  QSSConfig &compileTargetIR(bool flag) {
    compileTargetIRFlag = flag;
    return *this;
  }
  bool shouldCompileTargetIR() const { return compileTargetIRFlag; }

  QSSConfig &bypassPayloadTargetCompilation(bool flag) {
    bypassPayloadTargetCompilationFlag = flag;
    return *this;
  }
  bool shouldBypassPayloadTargetCompilation() const {
    return bypassPayloadTargetCompilationFlag;
  }

  QSSConfig &setPassPlugins(std::vector<std::string> plugins) {
    dialectPlugins = std::move(plugins);
    return *this;
  }
  const std::vector<std::string> &getPassPlugins() { return dialectPlugins; }

  QSSConfig &setDialectPlugins(std::vector<std::string> plugins) {
    dialectPlugins = std::move(plugins);
    return *this;
  }
  const std::vector<std::string> &getDialectPlugins() { return dialectPlugins; }

public:
  /// @brief Emit the configuration to stdout.
  void emit(llvm::raw_ostream &out) const;

protected:
  /// @brief input source (file path or direct input) to compile
  std::string inputSource = "-";
  /// @brief Whether inputSource directly contains the input source (otherwise
  /// it is a file path).
  bool directInputFlag = false;
  /// @brief Output path for the compiler output if emitting to file.
  std::string outputFilePath = "-";
  /// @brief The TargetSystem to target compilation for.
  std::optional<std::string> targetName = std::nullopt;
  /// @brief The path to the TargetSystem configuration information.
  std::optional<std::string> targetConfigPath = std::nullopt;
  /// @brief Source input type
  InputType inputType = InputType::None;
  /// @brief Output action to take
  EmitAction emitAction = EmitAction::None;
  /// @brief Register target passes with the compiler.
  bool addTargetPassesFlag = true;
  /// @brief Should available targets be printed
  bool showTargetsFlag = false;
  /// @brief Should available payloads be printed
  bool showPayloadsFlag = false;
  /// @brief Should the current configuration be printed
  bool showConfigFlag = false;
  /// @brief Should the plaintext payload be emitted
  bool emitPlaintextPayloadFlag = false;
  /// @brief Should the input source be included in the payload
  bool includeSourceFlag = false;
  /// @brief Should the IR be compiled for the target
  bool compileTargetIRFlag = false;
  /// @brief Should target payload generation be bypassed
  bool bypassPayloadTargetCompilationFlag = false;
  /// @brief Pass plugin paths
  std::vector<std::string> passPlugins;
  /// @brief Dialect plugin paths
  std::vector<std::string> dialectPlugins;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const QSSConfig &config);
std::ostream &operator<<(std::ostream &os, const QSSConfig &config);

/// @brief Assign the input configuration to be managed by the context.
/// @param context The context to assign the configuration to.
/// This must outlive all usages of the context registry.
/// @param config The configuration to move for the context.
void setContextConfig(mlir::MLIRContext *context, const QSSConfig &config);

/// @brief Get a constant reference to the configuration registered for this
/// context.
/// @param context The context to lookup the configuration for.
llvm::Expected<const QSSConfig &> getContextConfig(mlir::MLIRContext *context);

/// @brief Load a dynamic dialect plugin
/// @param pluginPath Path to the plugin
/// @param registry Dialect registry to register the plugin dialect with
mlir::LogicalResult loadDialectPlugin(const std::string &pluginPath,
                                      mlir::DialectRegistry &registry);

/// @brief Load a dynamic pass plugin
/// @param pluginPath Path to the plugin
mlir::LogicalResult loadPassPlugin(const std::string &pluginPath);

/// @brief A builder class for the QSSConfig. All standard configuration
/// population should be completed through builders.
class QSSConfigBuilder {
public:
  /// Build a new QSSConfig just from this builder
  virtual llvm::Expected<QSSConfig> buildConfig();
  /// Populate an existing QSSConfig from this builder.
  /// This may layer on top of existing configuration settings.
  virtual llvm::Error populateConfig(QSSConfig &config) = 0;
  virtual ~QSSConfigBuilder() = default;
};

/// Build the default tool configuration
/// @brief Build the QSSConfig using the standard sources and assign to the
/// supplied context.
///
/// The configuration precedence order is
/// 1. Default values
/// 2. Environment variables
/// 3. CLI arguments.
///
/// @return The constructed configuration
llvm::Expected<qssc::config::QSSConfig> buildToolConfig();

} // namespace qssc::config
#endif // QSS_QSSCONFIG_H
