//===- QUIROps.cpp - QUIR dialect attributes --------------------*- C++ -*-===//
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

#include "Dialect/QUIR/IR/QUIRAttributes.h"
#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIRTypes.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include <regex>

using namespace mlir;
using namespace mlir::quir;

uint64_t mlir::quir::DurationAttr::getSchedulingCycles(const double dt){
    double duration = getDuration().getValue().convertToDouble();

    // Convert through int64_t first to handle platform dependence
    int64_t converted;
    auto type = getType().dyn_cast<DurationType>();

    switch (type.getUnits()) {
    case TimeUnits::dt:
      converted = duration;
      break;
    case TimeUnits::fs:
      converted = duration / (1e15 * dt);
      break;
    case TimeUnits::ps:
      converted = duration / (1e12 * dt);
      break;
    case TimeUnits::ns:
      converted = duration / (1e9 * dt);
      break;
    case TimeUnits::us:
      converted = duration / (1e6 * dt);
      break;
    case TimeUnits::ms:
      converted = duration / (1e3 * dt);
      break;
    case TimeUnits::s:
      converted = duration / dt;
      break;
    }

    return static_cast<uint64_t>(converted);
}

