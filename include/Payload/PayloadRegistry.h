//===- PayloadRegistry.h - Payload Registry ---------------------*- C++ -*-===//
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
//
//  Declaration of the QSSC payload registry system.
//
//===----------------------------------------------------------------------===//
#ifndef PAYLOADREGISTRY_H
#define PAYLOADREGISTRY_H

#include "Payload.h"

#include "Plugin/PluginInfo.h"
#include "Plugin/PluginRegistry.h"

namespace qssc::payload::registry {

using PayloadInfo = plugin::registry::PluginInfo<Payload>;
using PayloadRegistry = plugin::registry::PluginRegistry<PayloadInfo>;

} // namespace qssc::payload::registry

#endif
