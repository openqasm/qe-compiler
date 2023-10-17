//===- ZipUtil.h ------------------------------------------------*- C++ -*-===//
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
/// Declares the Zip Utilities
///
//===----------------------------------------------------------------------===//

#ifndef PAYLOAD_ZIPUTIL_H
#define PAYLOAD_ZIPUTIL_H

#include <zip.h>

namespace qssc::payload {

// read zip into buffer - buffer allocated in function
char *read_zip_src_to_buffer(zip_source_t *zip_src, zip_int64_t &sz);

} // namespace qssc::payload

#endif // PAYLOAD_ZIPUTIL_H
