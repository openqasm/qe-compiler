//===- PayloadRegistry.h - Payload Registry ---------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2023.
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
