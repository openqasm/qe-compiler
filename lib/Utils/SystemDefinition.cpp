//===- SystemDefinition.cpp --------------------------------------*- C++-*-===//
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
#include "Utils/SystemNodes.h"
#include <initializer_list>
#include <ostream>
#include <sstream>
#include <string>
#include <unordered_map>

namespace qssc::utils {

bool SystemDefinition::hasValue() const { return initialized_; }

void SystemDefinition::plot(std::ostream &output) const { graph_.plot(output); }

std::string concat(std::initializer_list<std::string> list) {
  static const std::string separator = "-";
  std::string result;
  for (const auto &i : list) {
    result += i;
    result += separator;
  }
  return result;
}

std::string Qubit::name() const {
  static constexpr auto typeName = "Qubit";
  return concat({typeName, std::to_string(id)});
}

std::string Gate::name() const {
  static constexpr auto typeName = "Gate";
  std::stringstream qubits;
  std::copy(qubits_.begin(), qubits_.end(),
            std::ostream_iterator<int>(qubits, " "));
  return concat({typeName, name_, qubits.str()});
}

std::string Gate::type() const { return name_; }

std::vector<int> Gate::qubits() const { return qubits_; }

std::string Port::name() const {
  static constexpr auto typeName = "Port";
  return concat({typeName, name_});
}

std::string Port::id() const { return name_; }

Operation::Type Operation::stringToType(const std::string &op) {
  using Type = Operation::Type;
  static const std::unordered_map<std::string, Type> operationTable = {
      {"acquire", Type::Acquire},
      {"delay", Type::Delay},
      {"fc", Type::FrameChange},
      {"parametric_pulse", Type::Play}};

  auto it = operationTable.find(op);
  if (it != operationTable.end())
    return it->second;
  return Type::None;
}

std::string Operation::name() const {
  return order_ ? std::to_string(order_.value()) : "";
}

uint32_t Operation::order() const { return order_.value(); }

void Operation::order(uint32_t value) { order_ = value; }

PlayOp::Shape PlayOp::stringToShape(const std::string &shape) {
  using Shape = PlayOp::Shape;
  static const std::unordered_map<std::string, Shape> operationTable = {
      {"drag", Shape::Drag},
      {"gaussian", Shape::Gaussian},
      {"gaussian_square", Shape::GaussianSquare},
      {"constant", Shape::Constant},
      {"sampled_pulse", Shape::SampledPulse}};

  auto it = operationTable.find(shape);
  if (it != operationTable.end())
    return it->second;
  return Shape::None;
}

} // namespace qssc::utils
