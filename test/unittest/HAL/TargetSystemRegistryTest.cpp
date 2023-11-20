//===- TargetSystemRegistryTest.cpp -----------------------------*- C++ -*-===//
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
/// \file
/// This file implements test cases for TargetSystemRegistry.
///
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "HAL/TargetSystemRegistry.h"

namespace {

TEST(TargetSystemRegistry, LookupMockTarget) {
  // As a compiler developer, I want to register and lookup targets by name.

  const char *mockName = "mock";

  EXPECT_TRUE(
      qssc::hal::registry::TargetSystemRegistry::pluginExists(mockName));

  auto targetInfoOpt =
      qssc::hal::registry::TargetSystemRegistry::lookupPluginInfo(mockName);
  EXPECT_TRUE(targetInfoOpt.has_value());

  auto *targetInfo = targetInfoOpt.getValue();

  ASSERT_NE(targetInfo, nullptr);
  EXPECT_EQ(targetInfo->getName(), mockName);
}

} // anonymous namespace
