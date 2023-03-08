//===- QUIRInterfaces.td - QUIR dialect interfaces -*- C++ -*-================//
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
        static llvm::Optional<Operation *> getNextQubitOp(Operation *op) {
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
        static std::tuple<llvm::Optional<OpClass>, std::set<uint32_t>> getNextQubitOpOfTypeWithQubits(Operation *op) {
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
        static std::tuple<llvm::Optional<OpClass>, std::set<uint32_t>> getNextOpOfTypeWithQubits(Operation *op) {
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
        static llvm::Optional<OpClass> getNextQubitOpOfType(Operation *op) {
            auto result = getNextQubitOpOfTypeWithQubits<OpClass>(op);
            return std::get<0>(result);
        }

        /// If the next (lexicographically) Qubit operation is the specififed
        /// type implementing this interface return it.
        template <typename OpClass>
        static llvm::Optional<OpClass> getNextQubitOpIsType(Operation *op) {
            auto nextOp = getNextQubitOp(op);
            if (auto nextOpValue = nextOp.hasValue())
                if (OpClass castOp = dyn_cast<OpClass>(nextOpValue))
                    return castOp;
            return llvm::None;
        }
    }];
}

#endif // QUIR_INTERFACES
