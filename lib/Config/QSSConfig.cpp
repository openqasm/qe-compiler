//===- QSSConfig.cpp - Config info --------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  This file implements standard handling of QSSConfiguration options.
//
//===----------------------------------------------------------------------===//

#include "Config/QSSConfig.h"

#include "llvm/ADT/DenseMap.h"

using namespace qssc::config;




/// Mapping of registered MLIRContext configurations.
/// QUESTION: Rather than a global registry it seems like it would be much better
/// to inherit the MLIRContext as QSSContext and set the configuration on this?
/// Alternatively the QSSContext could own the MLIRContext?
static llvm::DenseMap<mlir::MLIRContext *, QSSConfig> contextConfigs{};


void qssc::config::setContextConfig(mlir::MLIRContext *context, QSSConfig config) {
    contextConfigs[context] = std::move(config);
}

llvm::Expected<const QSSConfig&> qssc::config::getContextConfig(mlir::MLIRContext *context) {
auto it = contextConfigs.find(context);
  if (it != contextConfigs.end())
    return it->getSecond();

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "Error: no config registered for the given context.\n");
}



llvm::Expected<QSSConfig> QSSConfigBuilder::buildConfig() {
    QSSConfig config;
    if (auto e = populateConfig(config))
        return e;
    return config;
}
