//===- PayloadV2.cpp --------------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
// Implements the PayloadV2 wrapper class
//
//===----------------------------------------------------------------------===//
#include <fcntl.h>
#include <vector>

#include <capnp/message.h>
#include <capnp/serialize-packed.h>

#include "Config.h"
#include "PayloadV2/PayloadV2.h"
#include "mock.capnp.h"

namespace fs = std::filesystem;

namespace qssc::payload {
void PayloadV2::writePlain(llvm::raw_ostream &stream) {
  // Copied from `ZipPayload::writePlain()`. Just using the produced files for
  // now, since it's the simplest (and we already get the files "for free").
  std::vector<fs::path> orderedNames = orderedFileNames();
  stream << "------------------------------------------\n";
  stream << "Plaintext payload: " << prefix << "\n";
  stream << "------------------------------------------\n";
  stream << "Manifest:\n";
  for (auto &fName : orderedNames)
    stream << fName << "\n";
  stream << "------------------------------------------\n";
  for (auto &fName : orderedNames) {
    stream << "File: " << fName << "\n";
    stream << files[fName];
    if (*(files[fName].rbegin()) != '\n')
      stream << "\n";
    stream << "------------------------------------------\n";
  }
}

void PayloadV2::writePlain(std::ostream &stream) {
  llvm::raw_os_ostream llstream(stream);
  this->writePlain(llstream);
}

auto encode(Component &component) -> kj::Own<capnp::MessageBuilder> {
  auto message = kj::heap<capnp::MallocMessageBuilder>();

  schema::Component::Builder builder = message->initRoot<schema::Component>();
  builder.setUid(component.uid);

  if (component.config.size() > 0) {
    auto el_size = sizeof(component.config[0]) / sizeof(uint8_t);
    builder.setConfig(
        kj::arrayPtr(reinterpret_cast<const uint8_t *>(component.config.data()),
                     el_size * component.config.size()));
  }

  if (component.program.size() > 0) {
    auto el_size = sizeof(component.config[0]) / sizeof(uint8_t);
    builder.setProgram(kj::arrayPtr(
        reinterpret_cast<const uint8_t *>(component.program.data()),
        el_size * component.program.size()));
  }

  return message;
}

auto encode(QuantumExecutionModule &qem) -> kj::Own<capnp::MessageBuilder> {
  auto message = kj::heap<capnp::MallocMessageBuilder>();

  schema::QuantumExecutionModule::Builder builder =
      message->initRoot<schema::QuantumExecutionModule>();

  capnp::List<schema::Component>::Builder components =
      builder.initComponents(qem.components.size());

  for (std::size_t i = 0; i < qem.components.size(); i++) {
    // Get reference to the input `QuantumExecutionModule` component
    const Component &component = qem.components[i];
    components[i].setUid(component.uid);

    // Only add the config if there's data
    if (component.config.size() > 0) {
      auto el_size = sizeof(component.config[0]) / sizeof(uint8_t);
      components[i].setConfig(kj::arrayPtr(
          reinterpret_cast<const uint8_t *>(component.config.data()),
          el_size * component.config.size()));
    }

    // Only add the program if there's data
    if (component.program.size() > 0) {
      auto el_size = sizeof(component.program[0]) / sizeof(uint8_t);
      components[i].setProgram(kj::arrayPtr(
          reinterpret_cast<const uint8_t *>(component.program.data()),
          el_size * component.program.size()));
    }

    // Multiply `sizeInWords()` by 8, as per word size defined by Cap'n Proto
    // (a "word" is defined as 8 bytes, or 64 bits).
    llvm::outs() << "Encoded component '" << components[i].getUid().cStr()
                 << "' (" << components[i].totalSize().wordCount * 8
                 << " bytes)\n";
  }

  llvm::outs() << "Encoded QuantumExecutionModule ("
               << message->sizeInWords() * 8 << " bytes)\n";

  return message;
}

Component PayloadV2::generate_component(hal::TargetInstrument &instrument) {
  /// Instantiate @c Component and initialize
  std::string name = instrument.getName();
  Component component{};
  component.uid = name;

  // Get the filename corresponding to the instrument name, if it exists
  auto orderedFNames = orderedFileNames();
  auto toFilename = [&](const std::string &name) -> std::string {
    auto it = std::find_if(
        orderedFNames.begin(), orderedFNames.end(), [&](const auto &fileName) {
          return fileName.generic_string().find(name) != std::string::npos;
        });
    return (it != orderedFNames.end()) ? it->generic_string() : "";
  };

  auto fname = toFilename(name);
  if (fname == "") {
    llvm::errs() << "No matching file for instrument name '" << name
                 << "'\n"
                    "unable to set configuration and program for instrument!";
  } else {
    // Get config data, and assign
    // TODO: Best way to access `SystemConfiguration` of a `TargetSystem`?

    // Get program data, and assign
    const auto &data = files[fname];
    component.program = std::vector<uint8_t>(data.begin(), data.end());
  }
  return component;
}

QuantumExecutionModule PayloadV2::generate_qem() {
  QuantumExecutionModule qem;

  // In general, a `TargetSystem` can have `TargetSystem` children. However,
  // the current minimalistic Cap'n Proto schema does not support this. For
  // now, and until we begin using multiple systems together, assume no
  // `TargetSystem` children, and consider only instruments.
  for (auto &instrument : target->getInstruments()) {
    auto component = this->generate_component(*instrument);
    qem.components.push_back(component);
  }

  return qem;
}

void PayloadV2::write_qem(llvm::raw_ostream &stream) {
  auto qem = this->generate_qem();
  auto encoded = encode(qem);

  if (name != "exp") {
    // Cap'n Proto works with file descriptors, but maybe there's a better way
    // to do this.
    auto fd = open(name.c_str(), O_WRONLY);
    if (fd == -1) {
      llvm::errs() << "Unable to open output file '" << name << "'!\n";
      return;
    }

    // If an output filename was provided, grab the corresponding file
    // descriptor for writing encoded data.
    llvm::outs() << "Writing QEM to as packed messaged\n";
    capnp::writePackedMessageToFd(fd, *encoded);
  } else {
    llvm::outs() << "------------------------------------------\n";
    llvm::outs() << "Writing QEM to stream\n";
    // No filename given, just write the QEM data to stream as usual.
    auto reader = encoded->getRoot<schema::QuantumExecutionModule>().asReader();
    for (auto it = reader.getComponents().begin();
         it != reader.getComponents().end(); it++) {
      llvm::outs() << "------------------------------------------\n";

      llvm::outs() << "Component: " << it->getUid() << "\n";
      auto config = it->getConfig();
      for (auto config_it = config.begin(); config_it != config.end();
           config_it++) {
        llvm::outs() << *config_it;
      }
      auto program = it->getProgram();
      for (auto program_it = program.begin(); program_it != program.end();
           program_it++) {
        llvm::outs() << *program_it;
      }
    }
  }
}

void PayloadV2::write_qem(std::ostream &stream) {
  llvm::raw_os_ostream llstream(stream);
  this->write_qem(llstream);
}

void PayloadV2::write(llvm::raw_ostream &stream) { this->write_qem(stream); }

void PayloadV2::write(std::ostream &stream) { this->write_qem(stream); }
} // namespace qssc::payload
