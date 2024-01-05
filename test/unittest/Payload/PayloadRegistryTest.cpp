//===- PayloadRegistryTest.cpp ----------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements test cases for PayloadRegistry.
///
//===----------------------------------------------------------------------===//

#include "gtest/gtest.h"

#include "Payload/PayloadRegistry.h"

namespace {

TEST(PayloadRegistry, LookupZipPayload) {
  // As a compiler developer, I want to register and lookup payloads by name.

  const char *zipName = "ZIP";

  EXPECT_TRUE(qssc::payload::registry::PayloadRegistry::pluginExists(zipName));

  auto payloadInfoOpt =
      qssc::payload::registry::PayloadRegistry::lookupPluginInfo(zipName);
  const bool payloadInfoPresent = payloadInfoOpt.has_value();
  EXPECT_TRUE(payloadInfoPresent);

  auto *payloadInfo = payloadInfoOpt.value();

  ASSERT_NE(payloadInfo, nullptr);
  EXPECT_EQ(payloadInfo->getName(), zipName);
}

} // anonymous namespace
