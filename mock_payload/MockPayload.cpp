
#include "MockPayload.h"

#include "Payload/PayloadRegistry.h"

using namespace qssc::payload::mock;

int qssc::payload::mock::init() {
  qssc::payload::registry::PayloadRegistration<MockPayload> registrar(
      "mock", "Mock payload for testing the payload infrastructure.",
      [](llvm::Optional<llvm::StringRef> configurationPath)-> llvm::Expected<std::unique_ptr<qssc::payload::Payload>> {
        return std::make_unique<MockPayload>();
      });
  return 0;
}


MockPayload::MockPayload() : Payload("prefix", "name") {}

void MockPayload::write(llvm::raw_ostream &stream) { };

void MockPayload::write(std::ostream &stream) {};

void MockPayload::writePlain(std::ostream &stream) {};

void MockPayload::writePlain(llvm::raw_ostream &stream) {};
