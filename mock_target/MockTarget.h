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

namespace qssc::targets::mock {

// Register the mock target.
int init();

class MockConfig : public qssc::hal::SystemConfiguration {
public:
  explicit MockConfig(llvm::StringRef configurationPath);
  auto getMultiplexingRatio() const -> uint { return multiplexing_ratio; }
  auto driveNode(uint qubitId) const -> uint { return qubitDriveMap[qubitId]; }
  auto acquireNode(uint qubitId) const -> uint {
    return qubitAcquireMap[qubitId];
  }
  auto acquireQubits(uint nodeId) -> const std::vector<int> & {
    return qubitAcquireToPhysIdMap[nodeId];
  }
  auto controllerNode() const -> uint { return controllerNodeId; }
  auto multiplexedQubits(uint qubitId) -> const std::vector<int> & {
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
  llvm::Error addPayloadPasses(mlir::PassManager &pm) override;
  auto payloadPassesFound(mlir::PassManager &pm) -> bool;
  llvm::Error addToPayload(mlir::ModuleOp &moduleOp,
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
  llvm::Error addPayloadPasses(mlir::PassManager &pm) override;
  llvm::Error addToPayload(mlir::ModuleOp &moduleOp,
                           payload::Payload &payload) override;

private:
  auto getModule(mlir::ModuleOp topModuleOp) -> mlir::ModuleOp;
  void buildLLVMPayload(mlir::ModuleOp &moduleOp, payload::Payload &payload);
}; // class MockController

class MockAcquire : public qssc::hal::TargetInstrument {
public:
  MockAcquire(std::string name, MockSystem *parent,
              const qssc::hal::SystemConfiguration &config);
  static void registerTargetPasses();
  static void registerTargetPipelines();
  llvm::Error addPayloadPasses(mlir::PassManager &pm) override;
  llvm::Error addToPayload(mlir::ModuleOp &moduleOp,
                           payload::Payload &payload) override;

private:
  auto getModule(mlir::ModuleOp topModuleOp) -> mlir::ModuleOp;
}; // class MockAcquire

class MockDrive : public qssc::hal::TargetInstrument {
public:
  MockDrive(std::string name, MockSystem *parent,
            const qssc::hal::SystemConfiguration &config);
  static void registerTargetPasses();
  static void registerTargetPipelines();
  llvm::Error addPayloadPasses(mlir::PassManager &pm) override;
  llvm::Error addToPayload(mlir::ModuleOp &moduleOp,
                           payload::Payload &payload) override;

private:
  auto getModule(mlir::ModuleOp topModuleOp) -> mlir::ModuleOp;
}; // class MockDrive

} // namespace qssc::targets::mock

#endif // HAL_TARGETS_MOCK_MOCKTARGET_H
