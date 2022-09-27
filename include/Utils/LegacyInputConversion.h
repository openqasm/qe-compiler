#ifndef UTILS_LEGACY_INPUT_CONVERSION_H
#define UTILS_LEGACY_INPUT_CONVERSION_H

#include "Utils/SystemDefinition.h"

#include "mlir/IR/Operation.h"

#include <string>

namespace qssc::utils {

class LegacyInputConversion : public SystemDefinition {

public:
  ~LegacyInputConversion() = default;
  LegacyInputConversion(mlir::Operation *op) {}
  void create(const std::string &calibrationsFilename,
              const std::string &expParamsFilename,
              const std::string &backendConfigFilename);
};

} // namespace qssc::utils

#endif // UTILS_LEGACY_INPUT_CONVERSION_H
