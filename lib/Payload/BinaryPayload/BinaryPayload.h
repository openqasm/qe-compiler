//===- BinaryPayload.h ------------------------------------------*- C++ -*-===//
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
///
/// Declares the BinaryPayload class
///
//===----------------------------------------------------------------------===//

#ifndef PAYLOAD_BINARYPAYLOAD_H
#define PAYLOAD_BINARYPAYLOAD_H

#include "Payload/Payload.h"

namespace qssc::payload {

// Register the binary payload.
int init_binary_payload();

class BinaryPayload : public Payload {
public:
  BinaryPayload() = default;
  BinaryPayload(PayloadConfig config) : Payload(std::move(config)) {}
  virtual ~BinaryPayload() = default;

  // write all files to the stream
  virtual void write(llvm::raw_ostream &stream) override;
  // write all files to the stream
  virtual void write(std::ostream &stream) override;
  // write all files in plaintext to the stream
  virtual void writePlain(std::ostream &stream) override;
  virtual void writePlain(llvm::raw_ostream &stream) override;
}; // class BinaryPayload

} // namespace qssc::payload

#endif // PAYLOAD_BINARYPAYLOAD_H
