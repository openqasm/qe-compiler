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

#include <string>

namespace qssc::hal::compile {

    class TargetCompilationScheduler {
        protected:
            TargetCompilationScheduler();

        public:
            virtual ~TargetCompilationScheduler() = default;
            virtual const std::string &getName() const;

    }; // class TargetCompilationScheduler

} // namespace qssc::hal::compile
#endif // TARGETCOMPILATIONSCHEDULER_H


