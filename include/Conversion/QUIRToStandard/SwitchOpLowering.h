//===- SwitchOpLowering.h ---------------------------------------*- C++ -*-===//
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
/// This file exposes patterns for lowering quir.switch.
///
//===----------------------------------------------------------------------===//

#ifndef QUIRTOSTD_SWITCHOPLOWERING_H
#define QUIRTOSTD_SWITCHOPLOWERING_H

namespace mlir::quir {

void populateSwitchOpLoweringPatterns(RewritePatternSet &patterns);

}; // namespace mlir::quir

#endif // QUIRTOSTD_SWITCHOPLOWERING_H
