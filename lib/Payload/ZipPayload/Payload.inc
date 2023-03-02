//===- Payload.inc - Mock payload registration ------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  This file defines static objects that register payloads
//  with the QSS compiler core.
//
//===----------------------------------------------------------------------===//

#ifndef PAYLOAD_PAYLOAD_ZIPPAYLOAD_H
#define PAYLOAD_PAYLOAD_ZIPPAYLOAD_H

#include "ZipPayload.h"

namespace qssc::payload {

[[maybe_unused]] int registrar = init();

} // namespace qssc::payload

#endif // PAYLOAD_PAYLOAD_ZIPPAYLOAD_H
