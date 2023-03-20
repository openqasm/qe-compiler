//===- Scheduling.cpp - Determine absolute timing in defcal's. ---*- C++-*-===//
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
///  This file implements the pass for filling in absolute timing attributes
///  within defcal calls.
///
//===----------------------------------------------------------------------===//

//#include "Dialect/Pulse/IR/PulseEnums.h"
//#include "Dialect/Pulse/Transforms/Scheduling.h"

// using namespace mlir;
// using namespace mlir::pulse;

// auto SchedulingPass::getResultHash(Operation *op) -> uint {
//  auto res = op->getResult(0);
//  return mlir::hash_value(res);
//}

// auto SchedulingPass::pulseCached(llvm::hash_code hash) -> bool {
//  if (pulseDurations.find(hash) != pulseDurations.end())
//    return true;
//  return false;
//}

// auto SchedulingPass::getFrameHashAndTime(mlir::Value &frame)
//    -> std::pair<uint, uint> {
//  auto *frameOp = frame.getDefiningOp();
//  auto frameHash = getResultHash(frameOp);
//  auto time = frameTimes[frameHash];
//  return std::make_pair(frameHash, time);
//}

// auto SchedulingPass::getMaxTime(mlir::OperandRange &frames) -> uint {
//  uint maxTime = 0;
//  for (auto frame : frames) {
//    // get frame time
//    auto pair = getFrameHashAndTime(frame);
//    auto time = pair.second;
//    if (time > maxTime)
//      maxTime = time;
//  }
//  return maxTime;
//}

///// templated waveform processing method
// template <class WaveformOp>
// void SchedulingPass::processOp(WaveformOp &wfrOp) {
//  // compute and cache waveform duration
//  auto wfrHash = getResultHash(wfrOp);
//  pulseDurations[wfrHash] = wfrOp.getDuration();
//}

// void SchedulingPass::processOp(Frame_CreateOp &frameOp) {
//  // initialize frame timing
//  auto frameHash = getResultHash(frameOp);
//  frameTimes[frameHash] = 0;
//}

// void SchedulingPass::processOp(DelayOp &delayOp) {
//  OpBuilder delayBuilder(delayOp); // delay operation builder
//  auto dur = delayOp.dur();        // operand
//  auto frames = delayOp.frames();
//  auto intDur = delayOp.getDuration(); // integer duration of delay

//  // split delays onto each frame and add timing attribute
//  for (auto frame : frames) {
//    // get frame hash and time
//    auto [frameHash, time] = getFrameHashAndTime(frame);
//    // create delay on frame w/ time tagged
//    auto frameDelayOp =
//        delayBuilder.create<DelayOp>(delayOp->getLoc(), dur, frame);
//    frameDelayOp->setAttr(
//        llvm::StringRef("t"),
//        delayBuilder.getIntegerAttr(delayBuilder.getI32Type(), time));
//    // update frame time
//    frameTimes[frameHash] += intDur;
//  }
//  // delete original op
//  delayOp.erase();
//}

// void SchedulingPass::processOp(BarrierOp &barrierOp) {
//  OpBuilder barrierBuilder(barrierOp); // barrier operation builder
//  auto frames = barrierOp.frames();
//  auto maxTime = getMaxTime(frames); // maximum time across frames

//  // more than one frame: compute max time among frames and
//  // add delays on all other frames to sync with this time
//  if (frames.size() > 1) {
//    for (auto frame : frames) {
//      // get frame hash and time
//      auto [frameHash, time] = getFrameHashAndTime(frame);
//      // create delays on non-max time frames
//      if (time < maxTime) {
//        auto len = maxTime - time;
//        // create length
//        auto lenOp = barrierBuilder.create<mlir::ConstantIntOp>(
//            barrierOp->getLoc(), len, barrierBuilder.getI32Type());
//        // create delay
//        auto delOp =
//            barrierBuilder.create<DelayOp>(barrierOp->getLoc(), lenOp, frame);
//        delOp->setAttr(
//            llvm::StringRef("t"),
//            barrierBuilder.getIntegerAttr(barrierBuilder.getI32Type(), time));
//        // update frame time
//        frameTimes[frameHash] += len;
//      }
//    }
//  }
//  // delete original op
//  barrierOp.erase();
//}

// void SchedulingPass::processOp(PlayOp &playOp) {
//  OpBuilder playBuilder(playOp); // play operation builder
//  // SSA form: waveform and frame should already be declared before
//  // play
//  auto *wfrOp = playOp.wfr().getDefiningOp();
//  auto *frameOp = playOp.frame().getDefiningOp();
//  auto wfrHash = getResultHash(wfrOp);
//  auto frameHash = getResultHash(frameOp);

//  // Add frame time as attribute
//  auto time = frameTimes[frameHash];
//  playOp->setAttr(llvm::StringRef("t"),
//                  playBuilder.getIntegerAttr(playBuilder.getI32Type(), time));

//  // Update frame time
//  auto wfrDur = pulseDurations[wfrHash];
//  frameTimes[frameHash] += wfrDur;
//}

// void SchedulingPass::processOp(CaptureOp &captureOp) {
//  OpBuilder captureBuilder(captureOp); // capture operation builder
//  // SSA form: frame should already be declared before capture
//  auto *frameOp = captureOp.frame().getDefiningOp();
//  auto frameHash = getResultHash(frameOp);

//  // Add frame time as attribute
//  auto time = frameTimes[frameHash];
//  captureOp->setAttr(
//      llvm::StringRef("t"),
//      captureBuilder.getIntegerAttr(captureBuilder.getI32Type(), time));

//  // Update frame time
//  auto capDur = captureOp.getDuration();
//  frameTimes[frameHash] += capDur;
//}

// void SchedulingPass::schedule(Operation *defCalOp) {
//  // add timing attributes for Pulse IR operations
//  defCalOp->walk([&](Operation *dcOp) {
//    if (auto sampWfrOp = dyn_cast<Waveform_CreateOp>(dcOp)) {
//      processOp(sampWfrOp);
//    } else if (auto gaussOp = dyn_cast<GaussianOp>(dcOp)) {
//      processOp(gaussOp);
//    } else if (auto gaussSqOp = dyn_cast<GaussianSquareOp>(dcOp)) {
//      processOp(gaussSqOp);
//    } else if (auto dragOp = dyn_cast<DragOp>(dcOp)) {
//      processOp(dragOp);
//    } else if (auto constWfrOp = dyn_cast<ConstWfrOp>(dcOp)) {
//      processOp(constWfrOp);
//    } else if (auto frameOp = dyn_cast<Frame_CreateOp>(dcOp)) {
//      processOp(frameOp);
//    } else if (auto delayOp = dyn_cast<DelayOp>(dcOp)) {
//      processOp(delayOp);
//    } else if (auto barrierOp = dyn_cast<BarrierOp>(dcOp)) {
//      processOp(barrierOp);
//    } else if (auto playOp = dyn_cast<PlayOp>(dcOp)) {
//      processOp(playOp);
//    } else if (auto captureOp = dyn_cast<CaptureOp>(dcOp)) {
//      processOp(captureOp);
//    }
//    // else: Pulse IR op does not impact scheduling
//  }); // defCalOp->walk
//}

///*** ASSUMPTIONS
// * Times are integers.
// * Barriers and delays are on frames, not qubits.
// * Stretches have been resolved.
// * No control flow.
// * Gate calls are resolved to defcal calls.
// ***/
///*** TODO PASSES : must occur before scheduling.
// * TODO: Write timing pass to update all quir lengths to std::constant
// integers
// *(in dt units).
// * TODO: Write pass to lower all barriers and delays onto frames from qubits.
// * TODO: Resolve stretch pass (should this be at scheduling time or earlier?)
// * TODO: Pass to handle control flow timing -> add mlir branching.
// * TODO: Pass to resolve gate calls to defcal calls (this should be a QUIR
// *pass).
// ***/
// void SchedulingPass::runOnOperation() {
//  // This pass is only called on the top-level module Op
//  Operation *moduleOperation = getOperation();
//  moduleOperation->walk([&](Operation *op) {
//    // find defcal call
//    if (auto callOp = dyn_cast<quir::CallDefCalGateOp>(op)) {
//      // find defcal body
//      auto calleeStr = callOp.getCallee();
//      // TODO: Lookup should include full function signature, not just the
//      // string
//      auto *defCalOp = SymbolTable::lookupSymbolIn(moduleOperation,
//      calleeStr); if (!defCalOp) {
//        callOp->emitError()
//            << "Could not find defcal body for " << calleeStr << ".";
//        return;
//      }

//      auto defCalHash = mlir::OperationEquivalence::computeHash(defCalOp);
//      // schedule this defcal if it has not already been scheduled
//      if (scheduledDefCals.find(defCalHash) == scheduledDefCals.end()) {
//        schedule(defCalOp);
//        // add defcal to those already scheduled
//        scheduledDefCals.insert(defCalHash);
//      }
//    }
//  }); // moduleOperation->walk
//} // runOnOperation
