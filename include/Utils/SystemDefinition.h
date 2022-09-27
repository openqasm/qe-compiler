//===- SystemDefinition.h --------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
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
