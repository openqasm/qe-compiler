//===- QUIRInterfaces.td - QUIR dialect interfaces -*- C++ -*-================//
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
///  Interfaces for the QUIR dialect
///
//===----------------------------------------------------------------------===//

#ifndef QUIR_INTERFACES
#define QUIR_INTERFACES

include "mlir/IR/OpBase.td"

// QUIR Interfaces

def QubitOpInterface : OpInterface<"QubitOpInterface"> {

    let description = [{
        Operation that applies to qubits and manipulates the quantum state.
    }];

    let cppNamespace = "::mlir::quir";

    let methods = [
        InterfaceMethod<
        /*desc=*/"Report the operated qubits for this operation",
        /*retTy=*/"::std::set<uint32_t>",
        /*methodName=*/"getOperatedQubits",
        /*args=*/(ins),
        /*methodBody=*/[{}],
        /*defaultImplementation=*/[{
            // By default walk the operation and find all nested operations
            // that implement QubitOpInterface
            return QubitOpInterface::getOperatedQubits($_op, true);
        }]
        >,
        InterfaceMethod<
        /*desc=*/"Get the qubits this operation shares qubits with another",
        /*retTy=*/"::std::set<uint32_t>",
        /*methodName=*/"getSharedQubits",
        /*args=*/(ins "Operation *":$other),
        /*methodBody=*/[{}],
        /*defaultImplementation=*/[{
            // By default walk the operation and find all nested operations
            // that implement QubitOpInterface
            return QubitOpInterface::getSharedQubits($_op, other);
        }]
        >,
        InterfaceMethod<
        /*desc=*/"Report if this operation shares qubits with another",
        /*retTy=*/"bool",
        /*methodName=*/"hasSharedQubits",
        /*args=*/(ins "Operation *":$other),
        /*methodBody=*/[{}],
        /*defaultImplementation=*/[{
            // By default walk the operation and find all nested operations
            // that implement QubitOpInterface
            return QubitOpInterface::opsShareQubits($_op, other);
        }]
        >,
    ];

    let extraSharedClassDeclaration = [{
        /// Returns the nested qubits operated on within the operation.
        static std::set<uint32_t> getOperatedQubits(mlir::Operation *op, bool ignoreSelf = false) {
           return interfaces_impl::getOperatedQubits(op, ignoreSelf);
        }

        /// Get the next (lexicographically) Qubit operation implementing this interface
        static std::optional<Operation *> getNextQubitOp(Operation *op) {
           return interfaces_impl::getNextQubitOp(op);
        }

        static std::set<uint32_t> getSharedQubits(std::set<uint32_t> &first, std::set<uint32_t> &second) {
           return interfaces_impl::getSharedQubits(first, second);
        }

        static std::set<uint32_t> getUnionQubits(std::set<uint32_t> &first, std::set<uint32_t> &second) {
           return interfaces_impl::getUnionQubits(first, second);
        }

        static std::set<uint32_t> getSharedQubits(mlir::Operation *first, mlir::Operation *second) {
           return interfaces_impl::getSharedQubits(first, second);
        }

        static bool opsShareQubits(mlir::Operation *first, mlir::Operation *second) {
           return interfaces_impl::opsShareQubits(first, second);
        }

        static bool qubitSetsOverlap(std::set<uint32_t> &first, std::set<uint32_t> &second) {
           return interfaces_impl::qubitSetsOverlap(first, second);
        }

        /// @brief Get the qubits between two operations. Not including the operations themselves
        static std::set<uint32_t> getQubitsBetweenOperations(mlir::Operation *first, mlir::Operation *second) {
           return interfaces_impl::getQubitsBetweenOperations(first, second);
        }

        /// Get the next (lexicographically) Qubit operation implementing this interface
        /// Accumulating the observed qubits along this path.
        template <typename OpClass>
        static std::tuple<std::optional<OpClass>, std::set<uint32_t>> getNextQubitOpOfTypeWithQubits(Operation *op) {
            Operation *curOp = op;
            std::set<uint32_t> operatedQubits;
            while (Operation *nextOp = curOp->getNextNode()) {
                if (isa<QubitOpInterface>(nextOp))
                    if (OpClass castOp = dyn_cast<OpClass>(nextOp))
                        return {castOp, operatedQubits};
                auto nextOpQubits = getOperatedQubits(nextOp);
                operatedQubits.insert(nextOpQubits.begin(), nextOpQubits.end());
                curOp = nextOp;
            }
            return {llvm::None, operatedQubits};
        }

        /// Get the next (lexicographically) operation (not necessarily of Qubit
        /// type) implementing this interface.
        /// Accumulating the observed qubits along this path.
        template <typename OpClass>
        static std::tuple<std::optional<OpClass>, std::set<uint32_t>> getNextOpOfTypeWithQubits(Operation *op) {
            Operation *curOp = op;
            std::set<uint32_t> operatedQubits;
            while (Operation *nextOp = curOp->getNextNode()) {
                if (OpClass castOp = dyn_cast<OpClass>(nextOp))
                    return {castOp, operatedQubits};
                auto nextOpQubits = getOperatedQubits(nextOp);
                operatedQubits.insert(nextOpQubits.begin(), nextOpQubits.end());
                curOp = nextOp;
            }
            return {llvm::None, operatedQubits};
        }

        /// Get the next (lexicographically) Qubit operation implementing this interface
        template <typename OpClass>
        static std::optional<OpClass> getNextQubitOpOfType(Operation *op) {
            auto result = getNextQubitOpOfTypeWithQubits<OpClass>(op);
            return std::get<0>(result);
        }

        /// If the next (lexicographically) Qubit operation is the specififed
        /// type implementing this interface return it.
        template <typename OpClass>
        static std::optional<OpClass> getNextQubitOpIsType(Operation *op) {
            auto nextOp = getNextQubitOp(op);
            if (auto nextOpValue = nextOp.hasValue())
                if (OpClass castOp = dyn_cast<OpClass>(nextOpValue))
                    return castOp;
            return llvm::None;
        }
    }];
}

#endif // QUIR_INTERFACES
