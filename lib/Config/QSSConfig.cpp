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

#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

#include <ostream>
#include <utility>

using namespace qssc::config;

// For now emit in a pseudo-TOML format.
void qssc::config::QSSConfig::emit(llvm::raw_ostream &os) {
  // Compiler configuration
  os << "[compiler]\n";
  os << "targetName: " << (getTargetName().has_value() ? getTargetName().value() : "None")
     << "\n";
  os << "targetConfigPath: "
     << (getTargetConfigPath().has_value() ? getTargetConfigPath().value() : "None")
     << "\n";
  os << "addTargetPasses: " << shouldAddTargetPasses() << "\n";
  os << "\n";

  // Mlir opt configuration
  os << "[opt]\n";
  os << "allowUnregisteredDialects: " << shouldAllowUnregisteredDialects() << "\n";
  os << "dumpPassPipeline: " << shouldDumpPassPipeline() << "\n";
  os << "emitBytecode: " << shouldEmitBytecode() << "\n";
  os << "bytecodeEmitVersion: " << bytecodeVersionToEmit() << "\n";
  os << "irdlFile: " << getIrdlFile() << "\n";
  os << "runReproducer: " << shouldRunReproducer() << "\n";
  os << "showDialects: " << shouldShowDialects() << "\n";
  os << "splitInputFile: " << shouldSplitInputFile() << "\n";
  os << "useExplicitModule" << shouldUseExplicitModule() << "\n";
  os << "verifyDiagnostics" << shouldVerifyDiagnostics() << "\n";
  os << "verifyPasses" << shouldVerifyPasses() << "\n";
  os << "verifyRoundTrip" << shouldVerifyRoundtrip() << "\n";
  os << "\n";

}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              qssc::config::QSSConfig &config) {
  config.emit(os);
  return os;
}

std::ostream &operator<<(std::ostream &os,
                         qssc::config::QSSConfig &config) {
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
