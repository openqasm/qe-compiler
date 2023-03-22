//===- PassRegistration.cpp -------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
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

#include "HAL/PassRegistration.h"
#include "HAL/TargetSystemRegistry.h"

using namespace qssc;
using namespace qssc::hal;

llvm::Error hal::registerTargetPasses() {
  llvm::Error err = llvm::Error::success();
  for (const auto &target :
       registry::TargetSystemRegistry::registeredPlugins()) {
    err =
        llvm::joinErrors(std::move(err), target.second.registerTargetPasses());
  }
  return err;
}

llvm::Error hal::registerTargetPipelines() {
  llvm::Error err = llvm::Error::success();
  for (const auto &target :
       registry::TargetSystemRegistry::registeredPlugins()) {
    err = llvm::joinErrors(std::move(err),
                           target.second.registerTargetPassPipelines());
  }
  return err;
}
