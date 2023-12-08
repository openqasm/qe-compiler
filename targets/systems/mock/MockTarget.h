//===- MockTarget.h - Mock target info --------------------------*- C++ -*-===//
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
//
//  This file declares the classes for the a Mock target interface
//
//===----------------------------------------------------------------------===//
#ifndef HAL_TARGETS_MOCK_MOCKTARGET_H
#define HAL_TARGETS_MOCK_MOCKTARGET_H

#include "HAL/SystemConfiguration.h"
#include "HAL/TargetSystem.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <unordered_map>

namespace qssc::targets::systems::mock {

// Register the mock target.
int init();

class MockConfig : public qssc::hal::SystemConfiguration {
public:
  explicit MockConfig(llvm::StringRef configurationPath);
  uint getMultiplexingRatio() const { return multiplexing_ratio; }
  uint driveNode(uint qubitId) const { return qubitDriveMap[qubitId]; }
  const std::vector<uint>& getDriveNodes() {
    return qubitDriveMap;
  }
  uint acquireNode(uint qubitId) const {
    return qubitAcquireMap[qubitId];
  }
  const std::vector<uint>& getAcquireNodes() {
    return qubitAcquireMap;
  }
  const std::vector<int> & acquireQubits(uint nodeId) {
    return qubitAcquireToPhysIdMap[nodeId];
  }
  uint controllerNode() const { return controllerNodeId; }
  const std::vector<int> & multiplexedQubits(uint qubitId) {
    return acquireQubits(acquireNode(qubitId));
  }

private:
  uint controllerNodeId;
  // The number of qubits attached to each acquire Mock
  uint multiplexing_ratio;
  std::vector<uint> qubitDriveMap;   // map from physId to drive NodeId
  std::vector<uint> qubitAcquireMap; // map from physId to acquire NodeId
  // map from acquire NodeId to a list of physical Ids that
  // this acquire node can capture from
  std::unordered_map<uint, std::vector<int>> qubitAcquireToPhysIdMap;
}; // class MockConfig

class MockSystem : public qssc::hal::TargetSystem {
public:
  static constexpr auto name = "mock";
  static const std::vector<std::string> childNames;
  explicit MockSystem(std::unique_ptr<MockConfig> config);
  static llvm::Error registerTargetPasses();
  static llvm::Error registerTargetPipelines();
  llvm::Error addPasses(mlir::PassManager &pm) override;
  auto payloadPassesFound(mlir::PassManager &pm) -> bool;
  llvm::Error emitToPayload(mlir::ModuleOp &moduleOp,
                           payload::Payload &payload) override;
  auto getConfig() -> MockConfig & { return *mockConfig; }

private:
  std::unique_ptr<MockConfig> mockConfig;
}; // class MockSystem

class MockController : public qssc::hal::TargetInstrument {
public:
  MockController(std::string name, MockSystem *parent,
                 const qssc::hal::SystemConfiguration &config);
  static void registerTargetPasses();
  static void registerTargetPipelines();
  virtual llvm::StringRef getNodeType() override { return "controller"; }
  // Currently there is a single controller with a fixed node id.
  virtual uint32_t getNodeId() override { return 1000; }
  llvm::Error addPasses(mlir::PassManager &pm) override;
  llvm::Error emitToPayload(mlir::ModuleOp &moduleOp,
                           payload::Payload &payload) override;

private:
  llvm::Error buildLLVMPayload(mlir::ModuleOp &moduleOp, payload::Payload &payload);
}; // class MockController

class MockAcquire : public qssc::hal::TargetInstrument {
public:
  MockAcquire(std::string name, MockSystem *parent,
              const qssc::hal::SystemConfiguration &config, uint32_t nodeId);
  static void registerTargetPasses();
  static void registerTargetPipelines();
  virtual llvm::StringRef getNodeType() override { return "acquire"; }
  virtual uint32_t getNodeId() override { return nodeId_; };
  llvm::Error addPasses(mlir::PassManager &pm) override;
  llvm::Error emitToPayload(mlir::ModuleOp &moduleOp,
                           payload::Payload &payload) override;

private:
  uint32_t nodeId_;

}; // class MockAcquire

class MockDrive : public qssc::hal::TargetInstrument {
public:
  MockDrive(std::string name, MockSystem *parent,
            const qssc::hal::SystemConfiguration &config, uint32_t nodeId);
  static void registerTargetPasses();
  static void registerTargetPipelines();
  virtual llvm::StringRef getNodeType() override { return "drive"; }
  virtual uint32_t getNodeId() override { return nodeId_; };
  llvm::Error addPasses(mlir::PassManager &pm) override;
  llvm::Error emitToPayload(mlir::ModuleOp &moduleOp,
                           payload::Payload &payload) override;

  private:
    uint32_t nodeId_;
}; // class MockDrive

} // namespace qssc::targets::systems::mock

#endif // HAL_TARGETS_MOCK_MOCKTARGET_H
