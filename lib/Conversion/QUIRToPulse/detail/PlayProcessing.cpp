//===- PlayProcessing.cpp - Pulse dialect ------------------C++-*-===//
//
// (C) Copyright IBM 2021 - 2023.
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

#include "Conversion/QUIRToPulse/QUIRToPulse.h"
#include "Dialect/Pulse/IR/PulseOps.h"
#include "Dialect/Pulse/IR/PulseTypes.h"
#include "Dialect/QUIR/IR/QUIROps.h"
#include "Utils/SystemNodes.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Value.h"

#include <algorithm>
#include <memory>
#include <vector>

namespace qu = qssc::utils;

namespace mlir::pulse {

void createDragPulse(const std::shared_ptr<qu::Drag> &pulse, Location loc,
                     MLIRContext *ctx, OpBuilder builder, Value target) {
  FloatType floatType = builder.getF64Type();
  IntegerType intType = builder.getI32Type();

  Value realAmp = builder.create<mlir::arith::ConstantOp>(
      loc, floatType, builder.getFloatAttr(floatType, pulse->amplitude.real()));

  Value imageAmp = builder.create<mlir::arith::ConstantOp>(
      loc, floatType, builder.getFloatAttr(floatType, pulse->amplitude.imag()));

  Value amp = builder.create<mlir::complex::CreateOp>(
      loc, ComplexType::get(FloatType::getF64(ctx)), realAmp, imageAmp);

  Value sigma = builder.create<mlir::arith::ConstantOp>(
      loc, intType, builder.getIntegerAttr(intType, pulse->sigma));

  Value beta = builder.create<mlir::arith::ConstantOp>(
      loc, floatType, builder.getFloatAttr(floatType, pulse->beta));

  Value duration = builder.create<mlir::arith::ConstantOp>(
      loc, intType, builder.getIntegerAttr(intType, pulse->duration));
  auto waveform = builder.create<DragOp>(loc, WaveformType::get(ctx), duration,
                                         amp, sigma, beta);

  builder.create<PlayOp>(loc, target, waveform);
}

void createGaussianSquarePulse(const std::shared_ptr<qu::GaussianSquare> &pulse,
                               Location loc, MLIRContext *ctx,
                               OpBuilder builder, Value target) {
  FloatType floatType = builder.getF64Type();
  IntegerType intType = builder.getI32Type();

  Value realAmp = builder.create<mlir::arith::ConstantOp>(
      loc, floatType, builder.getFloatAttr(floatType, pulse->amplitude.real()));

  Value imageAmp = builder.create<mlir::arith::ConstantOp>(
      loc, floatType, builder.getFloatAttr(floatType, pulse->amplitude.imag()));

  Value amp = builder.create<mlir::complex::CreateOp>(
      loc, ComplexType::get(FloatType::getF64(ctx)), realAmp, imageAmp);

  Value sigma = builder.create<mlir::arith::ConstantOp>(
      loc, intType, builder.getIntegerAttr(intType, pulse->sigma));

  Value width = builder.create<mlir::arith::ConstantOp>(
      loc, intType, builder.getIntegerAttr(intType, pulse->width));

  Value duration = builder.create<mlir::arith::ConstantOp>(
      loc, intType, builder.getIntegerAttr(intType, pulse->duration));

  auto waveform = builder.create<GaussianSquareOp>(loc, WaveformType::get(ctx),
                                                   duration, amp, sigma, width);

  builder.create<PlayOp>(loc, target, waveform);
}

void createGaussianPulse(const std::shared_ptr<qu::Gaussian> &pulse,
                         Location loc, MLIRContext *ctx, OpBuilder builder,
                         Value target) {
  FloatType floatType = builder.getF64Type();
  IntegerType intType = builder.getI32Type();

  Value realAmp = builder.create<mlir::arith::ConstantOp>(
      loc, floatType, builder.getFloatAttr(floatType, pulse->amplitude.real()));

  Value imageAmp = builder.create<mlir::arith::ConstantOp>(
      loc, floatType, builder.getFloatAttr(floatType, pulse->amplitude.imag()));

  Value amp = builder.create<mlir::complex::CreateOp>(
      loc, ComplexType::get(FloatType::getF64(ctx)), realAmp, imageAmp);

  Value sigma = builder.create<mlir::arith::ConstantOp>(
      loc, intType, builder.getIntegerAttr(intType, pulse->sigma));

  Value duration = builder.create<mlir::arith::ConstantOp>(
      loc, intType, builder.getIntegerAttr(intType, pulse->duration));

  auto waveform = builder.create<GaussianOp>(loc, WaveformType::get(ctx),
                                             duration, amp, sigma);

  builder.create<PlayOp>(loc, target, waveform);
}

void processPlayOps(const std::shared_ptr<qu::PlayOp> &play, Location loc,
                    MLIRContext *ctx, Value target, OpBuilder builder) {
  switch (play->shape) {
  case qu::PlayOp::Shape::Drag: {
    const auto pulse = std::dynamic_pointer_cast<qu::Drag>(play);

    if (!pulse) {
      llvm::errs() << "Encountered an unexpected error. pulse shape and "
                      "pointer mismatch";
    }
    createDragPulse(pulse, loc, ctx, builder, target);
    break;
  }
  case qu::PlayOp::Shape::GaussianSquare: {
    const auto pulse = std::dynamic_pointer_cast<qu::GaussianSquare>(play);

    if (!pulse) {
      llvm::errs() << "Encountered an unexpected error. pulse shape and "
                      "pointer mismatch";
    }
    createGaussianSquarePulse(pulse, loc, ctx, builder, target);
    break;
  }
  case qu::PlayOp::Shape::Gaussian: {
    const auto pulse = std::dynamic_pointer_cast<qu::Gaussian>(play);

    if (!pulse) {
      llvm::errs() << "Encountered an unexpected error. pulse shape and "
                      "pointer mismatch";
    }
    createGaussianPulse(pulse, loc, ctx, builder, target);
    break;
  }
  case qu::PlayOp::Shape::SampledPulse:
  case qu::PlayOp::Shape::Constant: {
    llvm::errs() << "Encountered unimplemented conversions.";
    break;
  }
  case qu::PlayOp::Shape::None: {
    llvm::errs() << "Encountered unexpected conversions.";
    break;
  }
  }
}
} // namespace mlir::pulse
