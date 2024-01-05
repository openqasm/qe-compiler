//===- ZipUtil.cpp ----------------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023, 2024.
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
/// Implements the Zip Utilities
///
//===----------------------------------------------------------------------===//

#include "ZipUtil.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <cstdlib>
#include <zip.h>
#include <zipconf.h>

char *qssc::payload::read_zip_src_to_buffer(zip_source_t *zip_src,
                                            zip_int64_t &sz) {
  //===---- Reopen for copying ----===//
  // reopen the archive stored in the new_archive_src
  zip_source_open(zip_src);
  // seek to the end of the archive
  zip_source_seek(zip_src, 0, SEEK_END);
  // get the number of bytes
  sz = zip_source_tell(zip_src);

  // allocate a new buffer to copy the archive into
  char *outbuffer = (char *)malloc(sz);
  if (!outbuffer) {
    llvm::errs()
        << "Unable to allocate output buffer for writing zip to stream\n";
    zip_source_close(zip_src);
    return nullptr;
  }

  // seek back to the begining of the archive
  zip_source_seek(zip_src, 0, SEEK_SET);
  // copy the entire archive into the output bufffer
  zip_source_read(zip_src, outbuffer, sz);
  // all done
  zip_source_close(zip_src);
  return outbuffer;
}
