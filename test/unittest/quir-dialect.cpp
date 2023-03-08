//===- quir-dialect.cpp - Unit tests for quir dialect -----------*- C++ -*-===//
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

#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Dialect/QUIR/Utils/Utils.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "gtest/gtest.h"

namespace {

class QUIRDialect : public ::testing::Test {
protected:
  mlir::MLIRContext ctx;
  mlir::UnknownLoc unkownLoc;
  mlir::ModuleOp rootModule;
  mlir::OpBuilder builder;

  QUIRDialect()
      : unkownLoc(mlir::UnknownLoc::get(&ctx)),
        rootModule(mlir::ModuleOp::create(unkownLoc)), builder(rootModule) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::quir::QUIRDialect>();
    ctx.appendDialectRegistry(registry);
    // since these tests are not built on passes, we cannot rely on MLIR's
    // infrastructure for loading dialects. thus, load them manually.
    ctx.loadAllAvailableDialects();
  }
};

TEST_F(QUIRDialect, CPTPOpTrait) {

  auto declareQubitOp = builder.create<mlir::quir::DeclareQubitOp>(
      unkownLoc, builder.getType<mlir::quir::QubitType>(1),
      builder.getIntegerAttr(builder.getI32Type(), 0));
  auto reset = builder.create<mlir::quir::ResetQubitOp>(
      unkownLoc, mlir::ValueRange{declareQubitOp.getResult()});

  EXPECT_FALSE(declareQubitOp->hasTrait<mlir::quir::UnitaryOp>());
  EXPECT_FALSE(declareQubitOp->hasTrait<mlir::quir::CPTPOp>());

  EXPECT_FALSE(reset->hasTrait<mlir::quir::UnitaryOp>());
  EXPECT_TRUE(reset->hasTrait<mlir::quir::CPTPOp>());

  EXPECT_FALSE(mlir::quir::isQuantumOp(declareQubitOp));
  EXPECT_TRUE(mlir::quir::isQuantumOp(reset));
}

TEST_F(QUIRDialect, UnitaryOpTrait) {

  auto declareQubitOp = builder.create<mlir::quir::DeclareQubitOp>(
      unkownLoc, builder.getType<mlir::quir::QubitType>(1),
      builder.getIntegerAttr(builder.getI32Type(), 0));
  auto barrier = builder.create<mlir::quir::BarrierOp>(
      unkownLoc, mlir::ValueRange{declareQubitOp.getResult()});

  EXPECT_TRUE(barrier->hasTrait<mlir::quir::UnitaryOp>());
  EXPECT_FALSE(barrier->hasTrait<mlir::quir::CPTPOp>());

  EXPECT_FALSE(mlir::quir::isQuantumOp(declareQubitOp));
  EXPECT_TRUE(mlir::quir::isQuantumOp(barrier));
}

TEST_F(QUIRDialect, MeasureSideEffects) {

  auto qubitDecl = builder.create<mlir::quir::DeclareQubitOp>(
      unkownLoc, builder.getType<mlir::quir::QubitType>(1),
      builder.getIntegerAttr(builder.getI32Type(), 0));

  auto measureOp = builder.create<mlir::quir::MeasureOp>(
      unkownLoc, builder.getI1Type(), qubitDecl.res());

  EXPECT_TRUE(measureOp);

  auto effectInterface =
      mlir::dyn_cast<mlir::MemoryEffectOpInterface>(measureOp.getOperation());

  ASSERT_TRUE(effectInterface);

  EXPECT_FALSE(mlir::isOpTriviallyDead(qubitDecl.getOperation()));
  EXPECT_FALSE(mlir::isOpTriviallyDead(measureOp.getOperation()));
}

} // namespace
