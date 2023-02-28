
#ifndef PAYLOADREGISTRY_H
#define PAYLOADREGISTRY_H

#include <string>

#include "llvm/ADT/Optional.h"
#include "llvm/Support/Error.h"
#include "llvm/ADT/StringMap.h"

#include "Payload.h"
#include "Support/Pimpl.h"

namespace qssc::payload::registry {

using PayloadFactoryFunction =
    std::function<llvm::Expected<std::unique_ptr<qssc::payload::Payload>>(
        llvm::Optional<llvm::StringRef> configurationPath)>;

class PayloadInfo {
public:
  /// Construct this entry
  PayloadInfo(llvm::StringRef name, llvm::StringRef description,
              PayloadFactoryFunction payloadFactory);
  ~PayloadInfo();
  /// Returns the name used to invoke this target from the CLI.
  llvm::StringRef getPayloadName() const { return name; }
  /// Returns the description for this target in the CLI.
  llvm::StringRef getPayloadDescription() const { return description; }
  /// Create the target system and register it under the given context.
  llvm::Expected<std::unique_ptr<Payload>>
  createPayload(llvm::Optional<llvm::StringRef> configurationPath);

  /// Print the help string for this Target.
  void printHelpStr(size_t indent, size_t descIndent) const;

private:
  /// The name of this Target to invoke from CLI.
  llvm::StringRef name;
  /// Description of this target.
  llvm::StringRef description;
  /// Target context factory function
  PayloadFactoryFunction payloadFactory;
};


void registerPayload(
    llvm::StringRef name, llvm::StringRef description,
    const PayloadFactoryFunction &targetFactory);

/// Look up the target info for a target. Returns None if not registered.
llvm::Optional<PayloadInfo *> lookupPayloadInfo(llvm::StringRef payloadName);

/// Verify the target exists
bool payloadExists(llvm::StringRef payloadName);

/// Available targets
const llvm::StringMap<PayloadInfo> &registeredPayloads();


template <typename ConcretePayload>
struct PayloadRegistration {
  PayloadRegistration(llvm::StringRef name,
                      llvm::StringRef description,
                      const PayloadFactoryFunction &payloadFactory) {
    registerPayload(name, description, payloadFactory);
  }
}; // struct PayloadRegistration

}; // namespace qssc::payload::registry

#endif // PAYLOADREGISTRY_H
