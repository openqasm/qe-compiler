//===- TargetCompilationScheduler.h - Compilation Scheduler -----*- C++ -*-===//
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
//  This file declares the classes for the top-level compilation scheduling
//  interface.
//
//===----------------------------------------------------------------------===//
#ifndef TARGETCOMPILATIONSCHEDULER_H
#define TARGETCOMPILATIONSCHEDULER_H


#include "HAL/TargetSystem.h"

#include "llvm/Support/Error.h"

#include <string>

using namespace qssc;

namespace qssc::hal::compile {

    /// @brief Base class for the compiler's
    /// target compilation infrastructure.
    /// A target system is a tree of compilation targets.
    /// We aim to support compiling each disjoint
    /// target subtree independently.
    class TargetCompilationScheduler {
        protected:
            TargetCompilationScheduler(hal::TargetSystem &target);

        public:
            virtual ~TargetCompilationScheduler() = default;
            virtual const std::string getName() const;

            /// @brief Get the base target system to be compiled.
            virtual hal::Target &getBaseTargetSystem() { return target; }

            /// @brief Compile only at the MLIR level for the full target
            /// system.
            /// @param moduleOp The root module operation to compile for.
            /// This must not be specialized to a system already.
            virtual llvm::Error addPasses(mlir::ModuleOp &moduleOp);

            /// @brief Generate the full configured compilation pipeline
            /// for all targets of the base target system. This will also
            /// invoke addPasses.
            /// @param moduleOp The root module operation to compile for.
            /// This must not be specialized to a system already.
            /// @param payload The payload to populate.
            virtual llvm::Error emitToPayload(mlir::ModuleOp &moduleOp, qssc::payload::Payload &payload);

        private:
            hal::TargetSystem &target;


    }; // class TargetCompilationScheduler

} // namespace hal::compile
#endif // TARGETCOMPILATIONSCHEDULER_H


