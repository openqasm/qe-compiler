//===- QuantumExecutionModule.h ---------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
// Declares the PayloadV2 wrapper class. This contents of these payloads are
// based on Cap'n Proto schemas.
//
//===----------------------------------------------------------------------===//

#ifndef QUANTUMEXECUTIONMODULE_QUANTUMEXECUTIONMODULE_H
#define QUANTUMEXECUTIONMODULE_QUANTUMEXECUTIONMODULE_H

#include <string>
#include <vector>

namespace qssc::payload {
/// @brief Container of target component (e.g., controller, AWG, etc.) data
/// Assume a simple base container for instruments: a unique ID to distinguish
/// instruments, a configuration, and program to execute.
struct Component {
  /// @brief Component unique identifier
  std::string uid;

  /// @brief Opaque data for a @c SystemConfiguration object
  /// Examples of configuration data could include the number of qubits in the
  /// system, multiplexing ratio, controller ID, etc.
  std::vector<uint8_t> config;

  /// @brief Opaque component program (execution) data
  std::vector<uint8_t> program;
};

/// @brief Container/primary payload data for target components
/// TODO: docs
struct QuantumExecutionModule {
  /// @brief Data of system components configuration
  std::vector<Component> components;
};

} // namespace qssc::payload

#endif // QUANTUMEXECUTIONMODULE_QUANTUMEXECUTIONMODULE_H
