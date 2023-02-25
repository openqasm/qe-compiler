#include "Payload/PayloadRegistry.h"

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/DenseMap.h>
#include "llvm/Support/ManagedStatic.h"

using namespace qssc::payload::registry;

static llvm::ManagedStatic<llvm::StringMap<PayloadInfo>> payloadRegistry;

PayloadInfo::PayloadInfo(
    llvm::StringRef name, llvm::StringRef description,
    PayloadFactoryFunction payloadFactory)
    : name(name), description(description),
      payloadFactory(std::move(payloadFactory)) {}

PayloadInfo::~PayloadInfo() = default;

llvm::Expected<std::unique_ptr<qssc::payload::Payload>>
PayloadInfo::createPayload(llvm::Optional<llvm::StringRef> configurationPath) {
  auto target = payloadFactory(configurationPath);
  if (!target)
    return target.takeError();
  return target;
}

/// Print the help information for this target. This includes the name,
/// description. `descIndent` is the indent that the
/// descriptions should be aligned.
void PayloadInfo::printHelpStr(size_t indent, size_t descIndent) const {
  size_t numSpaces = descIndent - indent - 4;
  llvm::outs().indent(indent)
      << "--" << llvm::left_justify(getPayloadName(), numSpaces) << "- "
      << getPayloadDescription() << '\n';
}


void registerPayload(
    const std::string& name, const std::string& description,
    const PayloadFactoryFunction &targetFactory) {

  payloadRegistry->try_emplace(name, name, description, targetFactory);
}

/// Look up the target info for a target. Returns None if not registered.
llvm::Optional<PayloadInfo *> lookupPayloadInfo(llvm::StringRef payloadName) {
  auto it = payloadRegistry->find(payloadName);
  if (it == payloadRegistry->end())
    return llvm::None;
  return &it->second;
}

/// Verify the target exists
bool payloadExists(llvm::StringRef payloadName) {
  auto it = payloadRegistry->find(payloadName);
  return it != payloadRegistry->end();
}

/// Available targets
const llvm::StringMap<PayloadInfo> &registeredPayloads() {
  return *payloadRegistry;
}
