//===- Scheduling.h - Add absolute timing to defcal calls. ------*- C++ -*-===//
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
///  This file declares the pass for adding absolute timing to defcal calls.
///
//===----------------------------------------------------------------------===//

//#ifndef PULSE_SCHEDULING_H
//#define PULSE_SCHEDULING_H

//#include <unordered_map>
//#include <unordered_set>

//#include "Dialect/Pulse/IR/PulseOps.h"
//#include "mlir/Pass/Pass.h"

// namespace mlir::pulse {

//// This pass applies absolute timing to each relevant Pulse IR instruction.
//// Timing is calculated on a per frame basis.
///*** Steps:
// * 1. Identify each defcal gate call.
// * 2. Find associated defcal body.
// * 3. Compute and store duration of each waveform and initialze time on each
// *frame. 4. For each play/delay instruction, increment the frame timing. Add
// *time attribute to instruction. For each barrier instruction, resolve to
// delays *on push forward basis (frames will be delayed to maximum time amongst
// all *frames).
// ***/
// struct SchedulingPass : public PassWrapper<SchedulingPass, OperationPass<>> {

//  std::unordered_set<uint>
//      scheduledDefCals; // hashes of defcal's that have already been scheduled
//  std::unordered_map<uint, uint>
//      pulseDurations; // mapping of waveform hashes to durations
//  std::unordered_map<uint, uint>
//      frameTimes; // mapping of frame hashes to time on that frame

//  // Hash an operation based on the result.
//  auto getResultHash(Operation *op) -> uint;

//  // Check if the pulse hash is cached in pulse durations.
//  // If it is cached, the hash will be found in pulseDurations.
//  auto pulseCached(llvm::hash_code hash) -> bool;

//  // Get hash and time of a frame as a std::pair
//  auto getFrameHashAndTime(mlir::Value &frame) -> std::pair<uint, uint>;

//  // Get the maximum time among a set of frames
//  auto getMaxTime(mlir::OperandRange &frames) -> uint;

//  // Process each operation in the defcal
//  template <class WaveformOp>
//  void processOp(WaveformOp &wfrOp);

//  void processOp(Frame_CreateOp &frameOp);

//  void processOp(DelayOp &delayOp);
//  void processOp(BarrierOp &barrierOp);
//  void processOp(PlayOp &playOp);
//  void processOp(CaptureOp &captureOp);

//  // Schedule the defcal
//  void schedule(Operation *defCalOp);

//  // Entry point for the pass
//  void runOnOperation() override;

//}; // end struct SchedulingPass
//} // namespace mlir::pulse

//#endif // PULSE_SCHEDULING_H
