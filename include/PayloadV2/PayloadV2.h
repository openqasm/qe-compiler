//===- PayloadV2.h ----------------------------------------------*- C++ -*-===//
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

#ifndef PAYLOADV2_PAYLOADV2_H
#define PAYLOADV2_PAYLOADV2_H

#include "Payload/Payload.h"
#include "PayloadV2/QuantumExecutionModule.h"

#include "HAL/SystemConfiguration.h"
#include "HAL/TargetSystem.h"

namespace qssc::payload {
// PayloadV2 class will wrap the QSS PayloadV2 and interface with the
// `qss-compiler`.
class PayloadV2 : public Payload {
public:
  /// @brief Default constructor.
  PayloadV2() = default;

  /// @brief Constructs a @c PaylaodV2 object for the given @c TargetSystem.
  /// @param target The @c TargetSystem to construct the payload for.
  PayloadV2(hal::TargetSystem *target) : target(target) {}

  /// @brief Construct a @c PayloadV2 given a @c TargetSystem and its
  ///        corresponding @c SystemConfiguration.
  /// @param target The @c TargetSystem to construct the payload for.
  /// @param config The target's @c SystemConfiguration.
  PayloadV2(hal::TargetSystem *target, hal::SystemConfiguration /* *config */)
      : target(target) {}

  /// @brief Construct a @c PayloadV2 given a @c TargetSystem, output file
  ///        prefix and filename.
  /// @param target The @c TargetSystem to construct the payload for.
  /// @param prefix Output file prefix.
  /// @param name Output filename.
  PayloadV2(hal::TargetSystem *target, std::string prefix, std::string name)
      : Payload(std::move(prefix), std::move(name)), target(target) {}
  virtual ~PayloadV2() = default;

  //===--------------------------------------------------------------------===//
  // Inherited functions
  //===--------------------------------------------------------------------===//

  /// @brief Write all data to `stream`
  /// For now, we will read data in from the files produced by a given target.
  /// Eventually, we will want to skip the file generation and write encoded
  /// data directly.
  virtual void write(llvm::raw_ostream &stream) override;
  virtual void write(std::ostream &stream) override;

  /// @brief Write all data to `stream`
  /// For now, we will read data in from the files produced by a given target.
  /// Eventually, we will want to skip the file generation and write encoded
  /// data directly.
  virtual void writePlain(llvm::raw_ostream &stream) override;
  virtual void writePlain(std::ostream &stream) override;

protected:
  /// @brief Generates component data.
  /// @param instrument The @c TargetInstrument from which to construct the
  ///        @c Component.
  /// @return @c Component of an instrument
  /// TODO: Get the @c SystemConfiguration of components.
  virtual Component generate_component(hal::TargetInstrument &instrument);

  /// @brief Generates quantum execution module (QEM).
  /// @return @c QuantumExecutionModule
  virtual QuantumExecutionModule generate_qem();

  /// @brief Writes a @c QuantumExecutionModule to the stream
  /// This function also provides a means to write a QEM to a file descriptor,
  /// as required per Cap'n Proto.
  /// @param stream The stream to write to.
  virtual void write_qem(llvm::raw_ostream &stream);

  /// @brief Writes a @c QuantumExecutionModule to the stream
  /// Effectively an alias that instantiates an @c llvm::raw_ostream from
  /// `stream` and calls the corresponding @c write_qem() function.
  /// @param stream The stream to write to.
  virtual void write_qem(std::ostream &stream);

  hal::TargetSystem *target;

private:
};
} // namespace qssc::payload

#endif // PAYLOADV2_PAYLOADV2_H
