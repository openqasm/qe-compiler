//===- GraphProcessing.h ---------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021 - 2023.
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
///
///  Utility functions for processing system graphs.
///
//===----------------------------------------------------------------------===//

#ifndef UTILS_GRAPH_PROCESSING_H
#define UTILS_GRAPH_PROCESSING_H

#include <boost/container/flat_map.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/depth_first_search.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/range/adaptor/transformed.hpp>

#include <map>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace qssc::utils {

class ObjectPropType {
public:
  virtual std::string name() const = 0;
  std::string uid() const {
    std::ostringstream address;
    address << this;
    return address.str();
  }
  virtual ~ObjectPropType() = default;
};

template <typename V, typename E>
class SystemGraph {

  using Graph =
      boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, V, E>;

public:
  using VertexDesc = typename Graph::vertex_descriptor;
  using EdgeDesc = typename Graph::edge_descriptor;

  void addEdge(const V src, const V dst, E edge = nullptr) {

    if (edge || edgeMap_.find(std::make_pair(src->uid(), dst->uid())) ==
                    edgeMap_.end()) {

      const auto srcIndex = addNode(src);
      const auto dstIndex = addNode(dst);
      const auto [ed, dup] = boost::add_edge(srcIndex, dstIndex, edge, graph_);

      edgeMap_.emplace(
          std::make_pair(graph_[srcIndex]->uid(), graph_[dstIndex]->uid()), ed);
      if (edge)
        edgeWeightMap_.emplace(edge->uid(), ed);
    }
  }

  template <typename T>
  std::vector<std::shared_ptr<T>> findAll() const {
    std::vector<std::shared_ptr<T>> results;

    for (const auto &p : vertexObjects())
      if (auto t = std::dynamic_pointer_cast<T>(p))
        results.emplace_back(t);
    return results;
  }

  template <typename T>
  std::vector<std::shared_ptr<T>>
  getEdgesFrom(const std::shared_ptr<ObjectPropType> &node) const {
    std::vector<std::shared_ptr<T>> results;
    const auto nodeIndex = vertexMap_.at(node->uid());

    for (const auto &e :
         boost::make_iterator_range(boost::out_edges(nodeIndex, graph_))) {
      const auto weight = graph_[e];
      if (weight) {
        if (auto t = std::dynamic_pointer_cast<T>(weight))
          results.emplace_back(t);
      }
    }
    return results;
  }

  template <typename T>
  std::vector<std::shared_ptr<T>>
  findAllFrom(const std::shared_ptr<ObjectPropType> &root) const {
    const auto rootIndex = vertexMap_.at(root->uid());
    std::map<VertexDesc, size_t> indexMap;
    for (auto v : boost::make_iterator_range(vertices(graph_)))
      indexMap.emplace(v, indexMap.size());
    const auto indexPropMap = boost::make_assoc_property_map(indexMap);
    std::vector<boost::default_color_type> colorMap(num_vertices(graph_));
    const auto colorPropMap =
        boost::make_iterator_property_map(colorMap.begin(), indexPropMap);

    std::vector<std::shared_ptr<T>> matches;
    auto matchesRef = std::ref(matches);

    struct SearchStopException : public std::exception {};

    const auto discoverFn = [&](VertexDesc v, const Graph &graph) {
      if (const auto p = std::dynamic_pointer_cast<T>(graph[v]))
        matchesRef.get().push_back(p);
    };

    const auto finishFn = [&](VertexDesc v, const Graph &graph) {
      if (v == rootIndex) {
        throw SearchStopException();
      }
    };

    dfs_visitor vis(discoverFn, finishFn);
    try {
      boost::depth_first_search(graph_, boost::visitor(vis)
                                            .vertex_index_map(indexPropMap)
                                            .color_map(colorPropMap)
                                            .root_vertex(rootIndex));
    } catch (const SearchStopException &) { // this is expected
    }

    return matches;
  }

  template <typename T>
  std::shared_ptr<T>
  getEdgeTarget(const std::shared_ptr<ObjectPropType> &edge) const {
    const auto ed = edgeWeightMap_.at(edge->uid());
    const auto target = boost::target(ed, graph_);
    return std::dynamic_pointer_cast<T>(graph_[target]);
  }

  void plot(std::ostream &out) const {
    const auto vb = boost::get(boost::vertex_bundle, graph_);
    const auto eb = boost::get(boost::edge_bundle, graph_);
    constexpr auto nothing = "";
    const auto label = [](auto bundle) {
      return boost::make_transform_value_property_map(
          [](const auto &prop) { return prop ? prop->name() : nothing; },
          bundle);
    };

    boost::write_graphviz(out, graph_, boost::make_label_writer(label(vb)),
                          boost::make_label_writer(label(eb)),
                          boost::default_writer{}, label(vb));
  }

private:
  struct dfs_visitor : boost::default_dfs_visitor {

    using Signature = std::function<void(VertexDesc, const Graph &)>;

    dfs_visitor(Signature discoverFn, Signature finishFn)
        : discoverFn_{std::move(discoverFn)}, finishFn_{std::move(finishFn)} {}

    void initialize_vertex(VertexDesc v, const Graph &g) const {}

    void start_vertex(VertexDesc v, const Graph &g) const {}

    void discover_vertex(VertexDesc v, const Graph &g) const {
      discoverFn_(v, g);
    }

    void finish_vertex(VertexDesc v, const Graph &g) const { finishFn_(v, g); }

  private:
    Signature discoverFn_, finishFn_;
  };

  VertexDesc addNode(const V node) {
    if (auto it = vertexMap_.find(node->uid()); it != vertexMap_.end())
      return it->second;

    const auto vd = boost::add_vertex(node, graph_);
    vertexMap_.emplace(graph_[vd]->uid(), vd);
    return vd;
  }

  auto vertexObjects() const {
    auto accessor = [map = boost::get(boost::vertex_bundle, graph_)](
        VertexDesc v) -> auto & {
      return map[v];
    };
    return boost::vertices(graph_) | boost::adaptors::transformed(accessor);
  }

private:
  std::map<std::string, VertexDesc> vertexMap_;
  std::map<std::string, EdgeDesc> edgeWeightMap_;
  std::map<std::pair<std::string, std::string>, EdgeDesc> edgeMap_;
  Graph graph_;
};

} // namespace qssc::utils

#endif
