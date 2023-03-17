//===- Target.inc - Mock target registration --------------------*- C++ -*-===//
//
// (C) Copyright IBM 2022, 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
//
//  This file defines static objects that register system targets
//  with the QSS compiler core.
//
//===----------------------------------------------------------------------===//

#ifndef HAL_TARGETS_MOCK_TARGET_H
#define HAL_TARGETS_MOCK_TARGET_H

#include "MockTarget.h"

namespace qssc::targets::mock {

[[maybe_unused]] int registrar = init();

} // namespace qssc::targets::mock

#endif // HAL_TARGETS_MOCK_TARGET_H
