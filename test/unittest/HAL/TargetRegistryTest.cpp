//===- TargetRegistryTest.cpp -----------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
