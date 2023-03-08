//===- SystemDefinition.h --------------------------------------*- C++ -*-===//
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
#ifndef UTILS_SYSTEM_DEFINITION_H
#define UTILS_SYSTEM_DEFINITION_H

#include "Utils/GraphProcessing.h"

#include <complex>
#include <cstdint>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace qssc::utils {

class SystemGraphVertex : public ObjectPropType {
public:
  ~SystemGraphVertex() = default;
};

class SystemGraphEdge : public ObjectPropType {
public:
  ~SystemGraphEdge() = default;
};

class SystemDefinition {

  using VertexProps = std::shared_ptr<SystemGraphVertex>;
  using EdgeProps = std::shared_ptr<SystemGraphEdge>;

public:
  virtual ~SystemDefinition() = default;

  template <typename T>
  std::vector<std::shared_ptr<T>> findAll() const {
    return graph_.findAll<T>();
  }

  template <typename T>
  std::vector<std::shared_ptr<T>>
  findAllFrom(std::shared_ptr<ObjectPropType> root) const {
    return graph_.findAllFrom<T>(root);
  }

  template <typename T>
  std::vector<std::shared_ptr<T>>
  getEdgesFrom(std::shared_ptr<ObjectPropType> node) const {
    return graph_.getEdgesFrom<T>(node);
  }

  template <typename T>
  std::shared_ptr<T> getEdgeTarget(std::shared_ptr<ObjectPropType> edge) const {
    return graph_.getEdgeTarget<T>(edge);
  }

  bool hasValue() const;

  void plot(std::ostream &output) const;

protected:
  SystemGraph<VertexProps, EdgeProps> graph_;
  bool initialized_ = false;
};

} // namespace qssc::utils

#endif // PULSE_SYSTEM_DEFINITION_H
