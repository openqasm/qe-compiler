//===- OQ3Dialect.cpp - OpenQASM 3 dialect ----------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file defines the OpenQASM 3 dialect in MLIR.
///
//===----------------------------------------------------------------------===//

#include "Dialect/OQ3/IR/OQ3Dialect.h"

// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/OQ3/IR/OQ3Ops.h"
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/OQ3/IR/OQ3Types.h"

#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::oq3;

/// Tablegen Definitions
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/OQ3/IR/OQ3OpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// OpenQASM 3 dialect
//===----------------------------------------------------------------------===//

// This class defines the interface for handling inlining with OQ3 operations.
// We simplify inherit from the base interface class and override the
// necessary methods.
struct OQ3InlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  // This hook checks to see if the given callable `Operation` is legal to
  // inline into the given call. For OQ3, this hook can simply return true, as
  // the OQ3 callable `Operation` is for now always inlinable.
  auto isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const -> bool final {
    return true;
  }

  // This hook checks to see if the given `Operation` is legal to inline into
  // the given region. For OQ3, this hook can simply return true, as all OQ3
  // operations are currently inlinable.
  auto isLegalToInline(Operation *, Region *, bool, IRMapping &) const
      -> bool final {
    return true;
  }
};

void OQ3Dialect::initialize() {

  addOperations<
#define GET_OP_LIST
// NOLINTNEXTLINE(misc-include-cleaner): Required for MLIR registrations
#include "Dialect/OQ3/IR/OQ3Ops.cpp.inc"
     >();

  addInterfaces<OQ3InlinerInterface>();
}
