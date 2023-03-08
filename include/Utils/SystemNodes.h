//===- SystemNodes.h - System graph nodes -----------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
#ifndef UTILS_SYSTEM_NODES_H
#define UTILS_SYSTEM_NODES_H

#include <Utils/SystemDefinition.h>
#include <complex>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace qssc::utils {

class Qubit : public SystemGraphVertex {
public:
  Qubit(uint32_t id) : SystemGraphVertex(), id(id) {}
  ~Qubit() = default;
  std::string name() const override;
  const uint32_t id;
};

class Gate : public SystemGraphVertex {
public:
  Gate(std::string name, std::vector<int> qubits)
      : name_{std::move(name)}, qubits_{std::move(qubits)} {}
  std::string name() const override;
  std::string type() const;
  std::vector<int> qubits() const;

private:
  const std::string name_;
  std::vector<int> qubits_;
};

class Port : public SystemGraphVertex {
public:
  Port(std::string name) : name_{std::move(name)} {}
  std::string name() const override;
  std::string id() const;

private:
  const std::string name_;
};

class Operation : public SystemGraphEdge {
public:
  Operation() : SystemGraphEdge(), order_{{}} {}
  ~Operation() = default;
  std::string name() const override;
  uint32_t order() const;
  void order(uint32_t value);

  enum class Type {
    Acquire,
    Delay,
    FrameChange,
    Play,
    None,
  };

  static Type stringToType(const std::string &op);

private:
  std::optional<uint32_t> order_;
};

class AcquisitionOp : public Operation {
public:
  ~AcquisitionOp() = default;
};

class DelayOp : public Operation {
public:
  DelayOp(uint32_t duration) : Operation(), duration{duration} {}
  ~DelayOp() = default;

  const uint32_t duration;
};

class FrameChangeOp : public Operation {
public:
  FrameChangeOp(double phase) : Operation(), phase{phase} {}
  ~FrameChangeOp() = default;

  const double phase;
};

class PlayOp : public Operation {
public:
  enum class Shape {
    Drag,
    Gaussian,
    GaussianSquare,
    Constant,
    SampledPulse,
    None,
  };

  PlayOp(const std::string &name, bool parametric, Shape shape)
      : Operation(), parametric{parametric}, shape{shape} {}

  ~PlayOp() = default;

  static Shape stringToShape(const std::string &shape);

  const bool parametric;
  const Shape shape;
  uint32_t t0() const;
  void t0(uint32_t value);

private:
  uint32_t t0_;
};

class Drag : public PlayOp {
public:
  static constexpr auto typeName = "Drag";
  Drag(std::complex<double> amplitude, double beta, uint32_t duration,
       uint32_t sigma)
      : PlayOp(typeName, true, PlayOp::Shape::Drag), amplitude{amplitude},
        beta{beta}, duration{duration}, sigma{sigma} {}
  ~Drag() = default;

  const std::complex<double> amplitude;
  const double beta;
  const uint32_t duration;
  const uint32_t sigma;
};

class GaussianSquare : public PlayOp {
public:
  static constexpr auto typeName = "GaussianSquare";
  GaussianSquare(std::complex<double> amplitude, uint32_t duration,
                 uint32_t sigma, uint32_t width)
      : PlayOp(typeName, true, PlayOp::Shape::GaussianSquare),
        amplitude{amplitude}, duration{duration}, sigma{sigma}, width{width} {}
  ~GaussianSquare() = default;

  const std::complex<double> amplitude;
  const uint32_t duration;
  const uint32_t sigma;
  const uint32_t width;
};

class Gaussian : public PlayOp {
public:
  static constexpr auto typeName = "Gaussian";
  Gaussian(std::complex<double> amplitude, uint32_t duration, uint32_t sigma,
           uint32_t width)
      : PlayOp(typeName, true, PlayOp::Shape::Gaussian), amplitude{amplitude},
        duration{duration}, sigma{sigma} {}
  ~Gaussian() = default;

  const std::complex<double> amplitude;
  const uint32_t duration;
  const uint32_t sigma;
};

class SampledPulse : public PlayOp {
public:
  static constexpr auto typeName = "SampledPulse";
  SampledPulse(std::vector<std::complex<double>> samples)
      : PlayOp(typeName, false, PlayOp::Shape::SampledPulse), samples{std::move(
                                                                  samples)} {}
  ~SampledPulse() = default;

  const std::vector<std::complex<double>> samples;
};

} // namespace qssc::utils

#endif // UTILS_SYSTEM_NODES_H
