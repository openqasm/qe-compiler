//===- Payload.h ------------------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
// Declares the Payload wrapper class
//
//===----------------------------------------------------------------------===//

#ifndef PAYLOAD_PAYLOAD_H
#define PAYLOAD_PAYLOAD_H

#include <filesystem>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "llvm/Support/raw_os_ostream.h"
#include "llvm/Support/raw_ostream.h"

namespace qssc {
  namespace payload {
// Payload class will wrap the QSS Payload and interface with the qss-compiler
class Payload {
public:
  Payload() : prefix(""), name("exp") {}
  Payload(std::string prefix, std::string name)
      : prefix(std::move(prefix) + "/"), name(std::move(name)) {
    files.clear();
  }
  virtual ~Payload() = default;

  // get/add the file fName and return a pointer to its data
  std::string *getFile(const std::string &fName);
  // get/add the file fName and return a pointer to its data
  std::string *getFile(const char *fName);
  // write all files to the stream
  virtual void write(llvm::raw_ostream &stream) = 0;
  // write all files to the stream
  virtual void write(std::ostream &stream) = 0;
  // write all files in plaintext to the stream
  virtual void writePlain(std::ostream &stream) = 0;
  virtual void writePlain(llvm::raw_ostream &stream) = 0;

  const std::string &getName() const { return name; }
  const std::string &getPrefix() const { return prefix; }

protected:
  // Class mutex
  std::mutex _mtx;

  // return an ordered list of filenames
  auto orderedFileNames() -> std::vector<std::filesystem::path>;

  // A hash function object to work with unordered_* containers:
  struct PathHash {
    std::size_t operator()(std::filesystem::path const &p) const noexcept {
      return std::filesystem::hash_value(p);
    }
  };

  std::string prefix;
  std::string name;
  std::unordered_map<std::filesystem::path, std::string, PathHash> files;
}; // class Payload

class ZipPayload : public Payload {
public:
  ZipPayload() = default;
  ZipPayload(std::string prefix, std::string name)
      : Payload(std::move(prefix), std::move(name)) {}
  virtual ~ZipPayload() = default;

  // write all files to the stream
  virtual void write(llvm::raw_ostream &stream) override;
  // write all files to the stream
  virtual void write(std::ostream &stream) override;
  // write all files in plaintext to the stream
  virtual void writePlain(std::ostream &stream) override;
  virtual void writePlain(llvm::raw_ostream &stream) override;

  // write all files to a zip archive named fName
  void writeZip(std::string fName);
  // write all files to a zip archive and output it to the stream
  void writeZip(std::ostream &stream);
  void writeZip(llvm::raw_ostream &stream);
  // write all files in plaintext to the dir named dirName
  void writePlain(const std::string &dirName = ".");

private:
  // creates a manifest json file
  void addManifest();

}; // class ZipPayload

}
} // namespace qssc::payload

#endif // PAYLOAD_PAYLOAD_H
