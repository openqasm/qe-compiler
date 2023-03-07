//===- QSSC.h ---------------------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022, 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file captures configuration and location of static and
/// temporary resources.
///
//===----------------------------------------------------------------------===//

#ifndef QSSC_H
#define QSSC_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>

namespace qssc {

namespace hal {
class Target;
} // namespace hal

/// Provide the version of the QSS compiler, which follows semantic versioning
/// (see https://semver.org/) and consists of major, minor, and patch version.
///
/// \returns version as StringRef
llvm::StringRef getQSSCVersion();

/// Provide the major version of the QSS compiler
///
/// \returns major version as StringRef
llvm::StringRef getQSSCMajorVersion();

/// Provide the minor version of the QSS compiler
///
/// \returns minor version as StringRef
llvm::StringRef getQSSCMinorVersion();

/// Provide the patch level of the QSS compiler
///
/// \returns patch level as StringRef
llvm::StringRef getQSSCPatchlevel();

/// Provide path to static resources (e.g., runtime libs). Note that targets
/// and other components must not make assumptions about the directory
/// hierarchy below this path.
///
/// \returns path to resources as StringRef
llvm::StringRef getResourcesDir();

/// Provides path to static resources for the given target. Targets must not
/// make assumptions about how this path relates to what getTargetResourcesDir
/// returns.
///
/// \param target the target for which to provide the path.
///
/// \returns path to static resources for target.
llvm::SmallString<128> getTargetResourcesDir(hal::Target const *target);

}; // namespace qssc

#endif // QSSC_H
