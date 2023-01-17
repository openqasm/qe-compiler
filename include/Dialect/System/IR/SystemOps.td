//===- SystemOps.td - System dialect ops -------------------*- tablegen -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#ifndef SYSTEM_OPS
#define SYSTEM_OPS

include "Dialect/System/IR/SystemInterfaces.td"
include "Dialect/System/IR/SystemAttributes.td"

include "Dialect/QUIR/IR/QUIRTypeConstraints.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

// TODO: This op should be extracted to a system-level dialect
def SYS_SystemInitOp : SYS_Op<"init", [IsolatedFromAbove]> {
    let summary = "Initializes the system";
    let description = [{
        The `sys.system_init` operation causes the system to be initialized,
        preparing for execution and synchronizing the system.
    }];
    let assemblyFormat = [{
        attr-dict
    }];
}

// TODO: This op should be extracted to a system-level dialect
def SYS_SystemFinalizeOp : SYS_Op<"finalize", [IsolatedFromAbove]> {
    let summary = "Finalizes the system";
    let description = [{
        The `sys.system_finalize` operation causes the system to be finalized,
        cleaning up results and shutting down execution.
    }];
    let assemblyFormat = [{
        attr-dict
    }];
}

#endif // SYSTEM_OPS
