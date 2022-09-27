//===- TargetRegistryTest.cpp -----------------------------------*- C++ -*-===//
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
/// This file implements test cases for TargetRegistry.
///
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "HAL/TargetRegistry.h"

namespace {

TEST(TargetRegistry, LookupMockTarget) {
  // As a compiler developer, I want to register and lookup targets by name.

  const char *mockName = "mock";

  EXPECT_TRUE(qssc::hal::registry::targetExists(mockName));

  auto targetInfoOpt = qssc::hal::registry::lookupTargetInfo(mockName);
  EXPECT_TRUE(targetInfoOpt.hasValue());

  auto *targetInfo = targetInfoOpt.getValue();

  ASSERT_NE(targetInfo, nullptr);
  EXPECT_EQ(targetInfo->getTargetName(), mockName);
}

} // anonymous namespace
