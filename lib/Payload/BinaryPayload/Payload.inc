//===- Payload.inc - Mock payload registration ------------------*- C++ -*-===//
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
///
/// This file defines static objects that register payloads
/// with the QSS compiler core.
///
//===----------------------------------------------------------------------===//

#ifndef PAYLOAD_PAYLOAD_BINARYPAYLOAD_H
#define PAYLOAD_PAYLOAD_BINARYPAYLOAD_H

#include "BinaryPayload.h"

namespace qssc::payload {

[[maybe_unused]] int bin_registrar = init_binary_payload();

} // namespace qssc::payload

#endif // PAYLOAD_PAYLOAD_BINARYPAYLOAD_H