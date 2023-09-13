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

//===----------------------------------------------------------------------===//
// Duration
//===----------------------------------------------------------------------===//

std::regex durationRegex("^([0-9]*[.]?[0-9]+)([a-zA-Z]*)");

llvm::Expected<Duration>
Duration::parseDuration(const std::string &durationStr) {
  std::smatch m;
  std::regex_match(durationStr, m, durationRegex);
  if (m.size() != 3)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        llvm::Twine("Unable to parse duration from ") + durationStr);

  double parsedDuration = std::stod(m[1]);
  // Convert all units to lower case.
  auto unitStr = m[2].str();
  auto lowerUnitStr = llvm::StringRef(unitStr).lower();
  DurationUnit parsedUnit;
  if (lowerUnitStr == "") {
    // Empty case is SI
    parsedUnit = DurationUnit::s;
  } else if (lowerUnitStr == "dt") {
    parsedUnit = DurationUnit::dt;
  } else if (lowerUnitStr == "ns") {
    parsedUnit = DurationUnit::ns;
  } else if (lowerUnitStr == "us") {
    parsedUnit = DurationUnit::us;
  } else if (lowerUnitStr == "ms") {
    parsedUnit = DurationUnit::ms;
  } else if (lowerUnitStr == "s") {
    parsedUnit = DurationUnit::s;
  } else {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   llvm::Twine("Unknown duration unit ") +
                                       unitStr);
  }

  return (Duration){.duration = parsedDuration, .unit = parsedUnit};
}

Duration Duration::convertToCycles(double dt) const {
  double convertedDuration;

  assert(unit == DurationUnit::dt || unit == DurationUnit::ns ||
         unit == DurationUnit::us || unit == DurationUnit::ms ||
         unit == DurationUnit::s);

  switch (unit) {
  case DurationUnit::dt:
    convertedDuration = duration;
    break;
  case DurationUnit::ns:
    convertedDuration = duration / (1e9 * dt);
    break;
  case DurationUnit::us:
    convertedDuration = duration / (1e6 * dt);
    break;
  case DurationUnit::ms:
    convertedDuration = duration / (1e3 * dt);
    break;
  case DurationUnit::s:
    convertedDuration = duration / dt;
    break;
  }
  return {convertedDuration, DurationUnit::dt};
}

//===----------------------------------------------------------------------===//
// DurationAttr
//===----------------------------------------------------------------------===//


Duration DurationAttr::getDuration() {
  // TODO: Should not be unwrapping Expected here
  auto durString = getDurationString().str();
  auto result = Duration::parseDuration(durString);
  if (auto E = result.takeError()) {
    llvm::errs() << "Error parsing duration " << durString << "\n";
    assert(false && "Error parsing duration");
  }
  return *result;
}
double DurationAttr::getValue() {
  return getDuration().duration;
}
Duration::DurationUnit DurationAttr::getUnits() {
  return getDuration().unit;
}
