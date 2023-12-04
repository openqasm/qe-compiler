//===- ThreadedCompilationScheduler.h - Threaded Scheduler ------*- C++ -*-===//
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
//
//  This file declares the classes for the top-level threaded compilation
//  interfaces.
//
//===----------------------------------------------------------------------===//
#ifndef THREADEDCOMPILATIONSCHEDULER_H
#define THREADEDCOMPILATIONSCHEDULER_H

#include "HAL/Compile/TargetCompilationScheduler.h"

#include "mlir/IR/MLIRContext.h"

#include <string>

namespace qssc::hal::compile {

    /// @brief A threaded implementation of a TargetCompilationScheduler
    /// based on the threading pools provided by the mlir::MLIRContext.
    /// This enables compilation across disjoint subtree of compilation
    /// targets in parallel. The implementation of parallel relies on MLIR's
    /// <a href="https://mlir.llvm.org/docs/PassManagement/#operation-pass">multi-threading assumptions</a>.
    /// As compilation is based on the shared MLIRContext's threadpool we are
    /// able to safely mix parallel nested passes and parallel target compilation
    /// subtrees without oversubscribing the compilation host's cores.
    class ThreadedCompilationScheduler : public TargetCompilationScheduler {
        protected:
            ThreadedCompilationScheduler(qssc::hal::TargetSystem &target, mlir::MLIRContext *context);

        public:
            virtual ~ThreadedCompilationScheduler() = default;
            virtual const std::string getName() const override;

            virtual llvm::Error compileMLIR(mlir::ModuleOp moduleOp) override;
            virtual llvm::Error compilePayload(mlir::ModuleOp moduleOp, qssc::payload::Payload &payload) override;

            mlir::MLIRContext* getContext() { return context;};
            llvm::ThreadPool& getThreadPool() {return getContext()->getThreadPool(); }

        private:
            /// Compiles the input module for a single target.
            llvm::Error compileMLIRTarget(Target &target, mlir::ModuleOp moduleOp);
            /// Compiles the input payload for a single target.
            llvm::Error compilePayloadTarget(Target &target, mlir::ModuleOp moduleOp, qssc::payload::Payload &payload);


            mlir::MLIRContext *context;



    }; // class THREADEDCOMPILATIONSCHEDULER

} // namespace qssc::hal::compile
#endif // THREADEDCOMPILATIONSCHEDULER_H
