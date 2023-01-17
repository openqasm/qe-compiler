//===- QSSC.cpp -------------------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements QSSC.
///
//===----------------------------------------------------------------------===//

#include "QSSC.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"

#include "Config.h"
// #include "Dialect/QUIR/IR/QUIRDialect.h"
#include "HAL/TargetSystem.h"

#include <cstdlib>
#include <string>

using namespace qssc;
using namespace llvm;

#define EXPORT_VERSION_STRING(FN, STR)                                         \
  llvm::StringRef qssc::FN() {                                                 \
    static const char *versionString = STR;                                    \
    return versionString;                                                      \
  }

EXPORT_VERSION_STRING(getQSSCMajorVersion, QSSC_VERSION_MAJOR)
EXPORT_VERSION_STRING(getQSSCMinorVersion, QSSC_VERSION_MINOR)
EXPORT_VERSION_STRING(getQSSCPatchlevel, QSSC_VERSION_PATCH)
EXPORT_VERSION_STRING(getQSSCVersion, QSSC_VERSION)

#undef EXPORT_VERSION_STRING

namespace {
llvm::StringRef _getResourcesDir() {
  if (char *env = getenv("QSSC_RESOURCES")) {
    /* strings returned by getenv may be invalidated, so keep a copy */
    static std::string resourcesDir{env};
    return resourcesDir;
  }

  /* fallback to compiled-in installation path */
  return QSSC_RESOURCES_INSTALL_PREFIX;
}

}; // namespace

llvm::StringRef qssc::getResourcesDir() {
  static llvm::StringRef resourcesDir = _getResourcesDir();
  return resourcesDir;
}

llvm::SmallString<128>
qssc::getTargetResourcesDir(qssc::hal::Target const *target) {
  // target-specific resources are at path "targets/<name of target>" below the
  // resource directory
  llvm::SmallString<128> path(getResourcesDir());

  llvm::sys::path::append(path, "targets", target->getName());
  return path;
}
