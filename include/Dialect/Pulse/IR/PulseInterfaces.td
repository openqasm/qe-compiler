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

    let methods = [
        InterfaceMethod<
        /*desc=*/"Get the timpoint that a pulse operation becomes effective",
        /*retTy=*/"::std::optional<int64_t>",
        /*methodName=*/"getTimepoint",
        /*args=*/(ins),
        /*methodBody=*/[{}],
        /*defaultImplementation=*/[{
            // By default, return the pulse.timepoint attribute
            return PulseOpSchedulingInterface::getTimepoint($_op);
        }]
        >,
        InterfaceMethod<
        /*desc=*/"Set the timpoint that a pulse operation becomes effective",
        /*retTy=*/"void",
        /*methodName=*/"setTimepoint",
        /*args=*/(ins "int64_t":$other),
        /*methodBody=*/[{}],
        /*defaultImplementation=*/[{
            // By default, set the pulse.timepoint attribute given an input arg
            return PulseOpSchedulingInterface::setTimepoint($_op, other);
        }]
        >,
        InterfaceMethod<
        /*desc=*/"Get the setup latency of a pulse operation",
        /*retTy=*/"::std::optional<uint64_t>",
        /*methodName=*/"getSetupLatency",
        /*args=*/(ins),
        /*methodBody=*/[{}],
        /*defaultImplementation=*/[{
            // By default, return the pulse.setupLatency attribute
            return PulseOpSchedulingInterface::getSetupLatency($_op);
        }]
        >,
        InterfaceMethod<
        /*desc=*/"Set the setup latency of a pulse operation",
        /*retTy=*/"void",
        /*methodName=*/"setSetupLatency",
        /*args=*/(ins "uint64_t":$other),
        /*methodBody=*/[{}],
        /*defaultImplementation=*/[{
            // By default, set the pulse.setupLatency attribute given an input arg
            return PulseOpSchedulingInterface::setSetupLatency($_op, other);
        }]
        >,
        InterfaceMethod<
        /*desc=*/"Get the total duration of a pulse operation",
        /*retTy=*/"::llvm::Expected<uint64_t>",
        /*methodName=*/"getDuration",
        /*args=*/(ins "Operation *":$other),
        /*methodBody=*/[{}],
        /*defaultImplementation=*/[{
            // By default, return the pulse.duration attribute
            return PulseOpSchedulingInterface::getDuration($_op, other);
        }]
        >,
        InterfaceMethod<
        /*desc=*/"Set the total duration of a pulse operation",
        /*retTy=*/"void",
        /*methodName=*/"setDuration",
        /*args=*/(ins "uint64_t":$other),
        /*methodBody=*/[{}],
        /*defaultImplementation=*/[{
            // By default, set the pulse.duration attribute given an input arg
            return PulseOpSchedulingInterface::setDuration($_op, other);
        }]
        >,
    ];

    let extraSharedClassDeclaration = [{
        static std::optional<int64_t> getTimepoint(mlir::Operation *op) {
            return interfaces_impl::getTimepoint(op);
        }

        static void setTimepoint(mlir::Operation *op, int64_t timepoint) {
            return interfaces_impl::setTimepoint(op, timepoint);
        }

        static std::optional<uint64_t> getSetupLatency(mlir::Operation *op) {
            return interfaces_impl::getSetupLatency(op);
        }

        static void setSetupLatency(mlir::Operation *op, uint64_t setupLatency) {
            return interfaces_impl::setSetupLatency(op, setupLatency);
        }

        static llvm::Expected<uint64_t> getDuration(mlir::Operation *op, mlir::Operation *callSequenceOp) {
            return interfaces_impl::getDuration(op, callSequenceOp);
        }

        static void setDuration(mlir::Operation *op, uint64_t duration) {
            return interfaces_impl::setDuration(op, duration);
        }

        // getDuration implementation for pulse operations with duration arg
        template <typename OpClass>
        static llvm::Expected<uint64_t> getDuration(Operation *op) {
            auto castOp = dyn_cast<OpClass>(op);
            auto durDeclOp = dyn_cast_or_null<mlir::arith::ConstantIntOp>(
                castOp.getDur().getDefiningOp());
            if (durDeclOp) {
                if (durDeclOp.value() < 0)
                return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                                "duration must be >= 0.");
                return durDeclOp.value();
            }
            return llvm::createStringError(
                llvm::inconvertibleErrorCode(),
                "Could not get the value of the op that defines duration!");
        }
    }];
}

#endif // PULSE_INTERFACES_TD
