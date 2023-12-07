//===- TargetCompilationScheduler.cpp ----------------------------*- C++ -*-===//
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

#include "HAL/Compile/TargetCompilationScheduler.h"

using namespace qssc::hal::compile;

TargetCompilationScheduler::TargetCompilationScheduler(qssc::hal::TargetSystem &target, mlir::MLIRContext *context) : target(target), context(context) {}

llvm::Error TargetCompilationScheduler::walkTarget(Target *target, mlir::ModuleOp targetModuleOp, WalkTargetFunction walkFunc) {
    // Call the input function for the walk on the target
    if (auto err = walkFunc(target, targetModuleOp))
        return err;

    for (auto *child : target->getChildren()) {
        // Recurse on the target
        if (auto err = walkTarget(child, walkFunc))
            return err;
    }
    return llvm::Error::success();

}
