//===- OQ3Types.h - OpenQASM 3 dialect types --------------*- C++ -*-=========//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This file declares the types in the OpenQASM 3 dialect.
///
//===----------------------------------------------------------------------===//

#ifndef DIALECT_OQ3_OQ3TYPES_H_
#define DIALECT_OQ3_OQ3TYPES_H_

// TODO: Temporary, until constraints between `OQ3`, `QUIR`, `Pulse`, and
// `QCS` dialects are ironed out.
#include "Dialect/QUIR/IR/QUIRTypes.h"

namespace mlir::oq3 {} // namespace mlir::oq3

#endif // DIALECT_OQ3_OQ3TYPES_H_
