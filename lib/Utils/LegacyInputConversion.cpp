//===- LegacyInputConversion.cpp - Converting legacy input  -----*- C++ -*-===//
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
#include "Utils/LegacyInputConversion.h"
#include "Utils/SystemDefinition.h"
#include "Utils/SystemNodes.h"

#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace qssc::utils {

auto verifyInput(const std::string &calibrationsFilename,
                 const std::string &expParamsFilename,
                 const std::string &backendConfigFilename) {

  std::ifstream calibrationConfigStream(calibrationsFilename);
  std::ifstream expParamsConfigStream(expParamsFilename);
  std::ifstream backendConfigStream(backendConfigFilename);

  if (!backendConfigStream)
    llvm::errs() << "Problem opening file " << backendConfigFilename << "\n";
  if (!expParamsConfigStream)
    llvm::errs() << "Problem opening file " << expParamsFilename << "\n";
  if (!calibrationConfigStream)
    llvm::errs() << "Problem opening file " << calibrationsFilename << "\n";

  nlohmann::json calibrationConfig;
  nlohmann::json expParamsConfig;
  nlohmann::json backendConfig;

  try {
    calibrationConfigStream >> calibrationConfig;
  } catch (std::exception &e) {
    llvm::errs() << e.what() << "\n";
  }

  try {
    expParamsConfigStream >> expParamsConfig;
  } catch (std::exception &e) {
    llvm::errs() << e.what() << "\n";
  }

  try {
    backendConfigStream >> backendConfig;
  } catch (std::exception &e) {
    llvm::errs() << e.what() << "\n";
  }

  return std::make_tuple(calibrationConfig, expParamsConfig, backendConfig);
}

auto createMaps(const nlohmann::json &backendConfig) {

  std::map<std::string, std::shared_ptr<qssc::utils::Port>> portMap;
  std::map<uint32_t, std::shared_ptr<qssc::utils::Qubit>> qubitMap;

  // create qubits
  {
    const auto &qubits = backendConfig["qubit_map"];
    if (qubits.size() < 1)
      llvm::errs() << "Unable to parse required field for declaring qubits.";
    for (const auto &i : qubits) {
      try {
        qubitMap[i] =
            std::make_shared<Qubit>(static_cast<uint32_t>(i.get<int>()));
      } catch (const nlohmann::json::parse_error &e) {
        llvm::errs() << "Unable to parse required field for qubit id.";
      }
    }
  }

  //   create ports
  {
    const auto &ports = backendConfig["channel_mappings"];
    if (ports.size() < 1)
      llvm::errs() << "Unable to parse required field for declaring ports.";
    for (const auto &[key, value] : ports.items()) {
      try {
        portMap[key] = std::make_shared<Port>(key);
      } catch (const nlohmann::json::parse_error &e) {
        llvm::errs() << "Unable to parse required field for port id.";
      }
    }
  }

  return std::make_tuple(qubitMap, portMap);
}

auto convertToComplex(const std::vector<double> &in) {
  return std::complex<double>(in[0], in[1]);
}

std::shared_ptr<PlayOp> selectPlayOp(
    nlohmann::json dict,
    std::map<std::string, std::vector<std::complex<double>>> pulse_library) {

  using Shape = PlayOp::Shape;
  std::shared_ptr<PlayOp> operation;

  auto shape = dict["pulse_shape"];
  auto name = dict["name"];
  if (shape.is_null())
    shape = "sampled_pulse";

  auto params = dict["parameters"];
  const auto amp = params["amp"];
  auto beta = params["beta"];
  auto duration = params["duration"];
  auto sigma = params["sigma"];
  auto width = params["width"];

  switch (PlayOp::stringToShape(shape)) {

  case Shape::Drag:
    operation =
        std::make_shared<Drag>(convertToComplex(amp), beta.get<double>(),
                               duration.get<int>(), sigma.get<int>());
    break;
  case Shape::Gaussian:
    operation =
        std::make_shared<Drag>(convertToComplex(amp), beta.get<double>(),
                               duration.get<int>(), sigma.get<int>());
    break;
  case Shape::GaussianSquare:
    operation = std::make_shared<GaussianSquare>(
        convertToComplex(amp), duration.get<int>(), sigma.get<int>(),
        width.get<int>());
    break;
  case Shape::Constant:
    operation =
        std::make_shared<Drag>(convertToComplex(amp), beta.get<double>(),
                               duration.get<int>(), sigma.get<int>());
    break;
  case Shape::SampledPulse:
    operation = std::make_shared<SampledPulse>(pulse_library[name]);
    break;
  case Shape::None:
    throw std::runtime_error("Encountered unexpected play type.");
    break;
  }
  return operation;
}

std::shared_ptr<Operation> selectOp(
    nlohmann::json dict,
    std::map<std::string, std::vector<std::complex<double>>> pulse_library) {

  const auto name = dict["name"].get<std::string>();
  std::shared_ptr<Operation> operation;
  std::string port;
  using Type = Operation::Type;

  switch (Operation::stringToType(name)) {

  case Type::Acquire: {
    operation = std::make_shared<AcquisitionOp>();
    break;
  }
  case Type::Delay: {
    const int duration = dict["duration"].get<int>();
    operation = std::make_shared<DelayOp>(duration);
    break;
  }
  case Type::Play: {
    operation = selectPlayOp(dict, pulse_library);
    break;
  }
  case Type::FrameChange: {
    auto phase = dict["phase"];
    if (phase.is_string())
      phase = 0.;
    operation = std::make_shared<FrameChangeOp>(phase);
    break;
  }
  case Type::None: {
    if (pulse_library.find(name) == pulse_library.end())
      throw std::runtime_error("Encountered unexpected operation type.");
    operation = selectPlayOp(dict, pulse_library);
    break;
  }
  }
  return operation;
};

void LegacyInputConversion::create(const std::string &calibrationsFilename,
                                   const std::string &expParamsFilename,
                                   const std::string &backendConfigFilename) {

  const auto [calibrationConfig, expParamsConfig, backendConfig] = verifyInput(
      calibrationsFilename, expParamsFilename, backendConfigFilename);

  auto [qubitMap, portMap] = createMaps(backendConfig);

  auto library = calibrationConfig["defaults"]["pulse_library"];
  std::map<std::string, std::vector<std::complex<double>>> pulse_library;
  if (!library.is_null()) {
    for (const auto &i : library) {
      const auto name = i["name"];
      const auto samples = i["samples"];
      std::vector<std::complex<double>> samplesArray;
      samplesArray.reserve(samples.size());
      for (const auto &s : samples)
        samplesArray.emplace_back(convertToComplex(s));
      pulse_library.emplace(name, samplesArray);
    }
  }

  const auto commands = calibrationConfig["defaults"]["cmd_def"];

  if (commands.is_null())
    llvm::errs() << "Unable to parse required field for commands look up.";

  if (commands.size() < 1)
    llvm::errs() << "Encountered unexpected empty commands field.";

  for (const auto &cmd : commands) {
    const auto name = cmd["name"].get<std::string>();
    const auto qubits = cmd["qubits"].get<std::vector<int>>();
    const auto &sequence = cmd["sequence"];
    uint32_t order = 0;

    const auto gate = std::make_shared<Gate>(name, qubits);

    for (const auto qubit : qubits)
      graph_.addEdge(qubitMap[qubit], gate);
    for (auto item : sequence) {

      const auto opName = item["name"].get<std::string>();
      const auto opLabel = item["label"];
      const auto portName = item["ch"];

      const auto port = portName.is_null() ? std::make_shared<Port>(opName)
                                           : portMap[portName];

      auto delay = item["t0"];
      if (delay.is_number_integer() && delay > 0) {
        const auto value = delay.get<uint32_t>();
        auto delayOp = std::make_shared<DelayOp>(value);
        delayOp->order(order);
        graph_.addEdge(gate, port, delayOp);
        ++order;
      }

      auto op = selectOp(item, pulse_library);
      op->order(order);

      graph_.addEdge(gate, port, op);
      ++order;
    }
  }
  initialized_ = true;
}

} // namespace qssc::utils
