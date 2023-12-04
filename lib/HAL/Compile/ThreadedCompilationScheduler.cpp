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

#include "HAL/Compile/ThreadedCompilationScheduler.h"

using namespace qssc::hal::compile;


namespace mlir {
    class ModuleOp;
}

ThreadedCompilationScheduler::ThreadedCompilationScheduler(qssc::hal::TargetSystem &target, mlir::MLIRContext *context) :  TargetCompilationScheduler(target), context(context) {}

const std::string ThreadedCompilationScheduler::getName() const { return "ThreadedCompilationScheduler"; }


llvm::Error ThreadedCompilationScheduler::compileMLIR(mlir::ModuleOp moduleOp) {


    return llvm::Error::success();
}


llvm::Error ThreadedCompilationScheduler::compileMLIRTarget(Target &target, mlir::ModuleOp moduleOp) {
    mlir::PassManager pm(getContext());
    if(auto err = target.addPasses(pm))
        return err;
    if(failed(pm.run(moduleOp)))
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                "Problems running the pass pipeline for target " + target.getName());
    return llvm::Error::success();

}


llvm::Error ThreadedCompilationScheduler::compilePayloadTarget(Target &target, mlir::ModuleOp moduleOp, qssc::payload::Payload &payload) {
    if (auto err = target.emitToPayload(moduleOp, payload))
      return err;
    return llvm::Error::success();
}
