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

#include <string>

namespace qssc::hal::compile {

    class ThreadedCompilationScheduler : public TargetCompilationScheduler {
        protected:
            ThreadedCompilationScheduler();

        public:
            virtual ~ThreadedCompilationScheduler() = default;
            virtual const std::string getName() const;

    }; // class THREADEDCOMPILATIONSCHEDULER

} // namespace qssc::hal::compile
#endif // THREADEDCOMPILATIONSCHEDULER_H
