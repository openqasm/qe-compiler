//===- PulseInterfaces.td - Pulse dialect interfaces -*- C++ -*-==============//
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
///  Interfaces for the Pulse dialect
///
//===----------------------------------------------------------------------===//

#ifndef PULSE_INTERFACES_TD
#define PULSE_INTERFACES_TD

include "mlir/IR/OpBase.td"

// Pulse Interfaces

def PulseOpSchedulingInterface : OpInterface<"PulseOpSchedulingInterface"> {

    let description = [{
        An interface to define functions related to scheduling of pulse operations.
    }];

    let cppNamespace = "::mlir::pulse";

    let extraSharedClassDeclaration = [{
        static int64_t getTimepoint(mlir::Operation *op) {
            return interfaces_impl::getTimepoint(op);
        }

        static void setTimepoint(mlir::Operation *op, IntegerAttr timepointAttr) {
            return interfaces_impl::setTimepoint(op, timepointAttr);
        }

        static int64_t getSetupLatency(mlir::Operation *op) {
            return interfaces_impl::getSetupLatency(op);
        }

        static void setSetupLatency(mlir::Operation *op, IntegerAttr setupLatencyAttr) {
            return interfaces_impl::setSetupLatency(op, setupLatencyAttr);
        }
    }];
}

#endif // PULSE_INTERFACES_TD
