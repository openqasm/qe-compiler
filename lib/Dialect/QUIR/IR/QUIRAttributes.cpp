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

using namespace mlir;
using namespace mlir::quir;


//===----------------------------------------------------------------------===//
// DurationAttr
//===----------------------------------------------------------------------===//

namespace {
  /// Convert input value (in units of inputUnits) to target units
  double convertToSeconds(const double value, const TimeUnits inputUnits, const double dt=1.) {
      switch (inputUnits) {
      case TimeUnits::dt:
        return value * dt;
        break;
      case TimeUnits::fs:
        return value / 1.e15;
        break;
      case TimeUnits::ps:
        return value / 1.e12;
        break;
      case TimeUnits::ns:
        return value / 1.e9;
        break;
      case TimeUnits::us:
        return value / 1.e6;
        break;
      case TimeUnits::ms:
        return value / 1.e3;
        break;
      case TimeUnits::s:
        return value;
        break;
      }
      llvm_unreachable("unhandled TimeUnits conversion.");
  }

  /// Convert input value (in units of seconds) to target outputUnits
  double convertFromSeconds(const double value, const TimeUnits outputUnits, const double dt=1.) {
      switch (outputUnits) {
        case TimeUnits::dt:
          return value / dt;
          break;
        case TimeUnits::fs:
          return value * 1.e15;
          break;
        case TimeUnits::ps:
          return value * 1.e12;
          break;
        case TimeUnits::ns:
          return value * 1.e9;
          break;
        case TimeUnits::us:
          return value * 1.e6;
          break;
        case TimeUnits::ms:
          return value * 1.e3;
          break;
        case TimeUnits::s:
          return value;
          break;
      }
      llvm_unreachable("unhandled TimeUnits conversion.");

  }


} // anonymous namespace


double DurationAttr::getDtFromSchedulingRate(const double schedulingRate) {
  return 1./schedulingRate;
}

double DurationAttr::getSchedulingRateFromDt(const double dt) {
  return 1./dt;
}

double DurationAttr::convertUnitsToUnits(double value, TimeUnits inputUnits, TimeUnits outputUnits, const double dt) {
  // Check if we can avoid the conversion.
  if (inputUnits == outputUnits)
    return value;
  double seconds = convertToSeconds(value, inputUnits, dt);
  return convertFromSeconds(seconds, outputUnits, dt);
}


uint64_t DurationAttr::getSchedulingCycles(const double dt) {
  double duration = convertUnits(TimeUnits::dt, dt);

  // Convert through int64_t first to handle platform dependence
  return static_cast<int64_t>(duration);

}

double DurationAttr::convertUnits(const TimeUnits targetUnits, const double dt) {
    double duration = getDuration().convertToDouble();
    return DurationAttr::convertUnitsToUnits(duration, getType().dyn_cast<DurationType>().getUnits(), targetUnits, dt);
}

DurationAttr DurationAttr::getConvertedDurationAttr(const TimeUnits targetUnits, const double dt) {
  return DurationAttr::get(getContext(), DurationType::get(getContext(), targetUnits), llvm::APFloat(convertUnits(targetUnits, dt)));
}

