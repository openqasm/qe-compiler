//===- TargetSystem.cpp -----------------------------------------*- C++ -*-===//
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

#include "llvm/ADT/SmallString.h"

#include "HAL/TargetSystem.h"

using namespace qssc::hal;
using namespace qssc::payload;

Target::Target(std::string name, Target *parent)
    : name(std::move(name)), parent(parent) {}

TargetSystem::TargetSystem(std::string name, Target *parent)
    : Target(std::move(name), parent) {}

TargetInstrument::TargetInstrument(std::string name, Target *parent)
    : Target(std::move(name), parent) {}
