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
    double duration = getDuration().convertToDouble();

    auto type = getType().dyn_cast<DurationType>();

    // Convert through int64_t first to handle platform dependence
    switch (type.getUnits()) {
    case TimeUnits::dt:
      return static_cast<int64_t>(duration);
      break;
    case TimeUnits::fs:
      return static_cast<int64_t>(duration / (1e15 * dt));
      break;
    case TimeUnits::ps:
      return static_cast<int64_t>(duration / (1e12 * dt));
      break;
    case TimeUnits::ns:
      return static_cast<int64_t>(duration / (1e9 * dt));
      break;
    case TimeUnits::us:
      return static_cast<int64_t>(duration / (1e6 * dt));
      break;
    case TimeUnits::ms:
      return static_cast<int64_t>(duration / (1e3 * dt));
      break;
    case TimeUnits::s:
      return static_cast<int64_t>(duration / dt);
      break;
    }
    llvm_unreachable("unhandled TimeUnits conversion.");
}

