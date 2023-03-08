//===- PassRegistration.cpp -------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#include "HAL/PassRegistration.h"
#include "HAL/TargetSystemRegistry.h"

using namespace qssc;
using namespace qssc::hal;

llvm::Error hal::registerTargetPasses() {
  llvm::Error err = llvm::Error::success();
  for (const auto &target : registry::TargetSystemRegistry::registeredPlugins()) {
    err =
        llvm::joinErrors(std::move(err), target.second.registerTargetPasses());
  }
  return err;
}

llvm::Error hal::registerTargetPipelines() {
  llvm::Error err = llvm::Error::success();
  for (const auto &target : registry::TargetSystemRegistry::registeredPlugins()) {
    err = llvm::joinErrors(std::move(err),
                           target.second.registerTargetPassPipelines());
  }
  return err;
}
