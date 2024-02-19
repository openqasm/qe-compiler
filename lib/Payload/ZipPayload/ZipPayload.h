//===- ZipPayload.h ---------------------------------------------*- C++ -*-===//
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
/// Declares the ZipPayload class
///
//===----------------------------------------------------------------------===//

#ifndef PAYLOAD_ZIPPAYLOAD_H
#define PAYLOAD_ZIPPAYLOAD_H

#include "Payload/Payload.h"

namespace qssc::payload {

// Register the zip payload.
int init();

class ZipPayload : public Payload {
public:
  ZipPayload() = default;
  ZipPayload(PayloadConfig config) : Payload(std::move(config)) {}
  virtual ~ZipPayload() = default;

  // write all files to the stream
  virtual void write(llvm::raw_ostream &stream) override;
  // write all files to the stream
  virtual void write(std::ostream &stream) override;
  // write all files in plaintext to the stream
  virtual void writePlain(std::ostream &stream) override;
  virtual void writePlain(llvm::raw_ostream &stream) override;
  virtual void
  writeArgumentSignature(qssc::arguments::Signature &&sig) override;

  // write all files to a zip archive named fName
  void writeZip(std::string fName);
  // write all files to a zip archive and output it to the stream
  void writeZip(std::ostream &stream);
  void writeZip(llvm::raw_ostream &stream);
  // write all files in plaintext to the dir named dirName
  void writePlain(const std::string &dirName = ".");
  void addFile(llvm::StringRef filename, llvm::StringRef str) override;

private:
  // creates a manifest json file
  void addManifest();

}; // class ZipPayload

} // namespace qssc::payload

#endif // PAYLOAD_ZIPPAYLOAD_H
