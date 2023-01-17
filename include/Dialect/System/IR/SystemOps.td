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

// Define a side effect that identifies an operation as not dead while not
// interfering with memory operations (e.g., allows store-forwarding across
// this operation).
// Note that operations without memory effects defined will be treated
// conservatively (i.e., not making any assumptions).
// see lib/Interfaces/SideEffectInterfaces.cpp:isTriviallyDeadImpl()
// see lib/Dialect/Affine/Utils/Utils.cpp:hasNoInterveningEffect()
def NonInterferingNonDeadSideEffect : MemoryEffects<[MemFree<DefaultResource>]>;

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

def SYS_ShotInitOp : SYS_Op<"shot_loop_init", [IsolatedFromAbove]> {
    let summary = "Initialization for shot loops";
    let description = [{
        The `sys.shot_loop_init` operation causes the system to initialize the shot
        loop to maintain consistent initial conditions for each shot
    }];
    let assemblyFormat = [{
        attr-dict
    }];
}

// def SYS_BroadcastOp : SYS_Op<"broadcast", [NonInterferingNonDeadSideEffect]> {
//     let summary = "Broadcast a value from this controller to all others";
//     let description = [{
//         The `quir.broadcast` operation represents a broadcast command that sends a value
//         from this controller to all others. All other controllers should have a
//         corresponding `quir.recv` operation. If only one controller should receive the
//         message then the `quir.send` operation should be used instead.

//         Example:
//         ```mlir
//         %angle1 = quir.constant #quir.angle<0.2 : !quir.angle<20>>
//         quir.broadcast %angle1 : !quir.angle<20>
//         ```
//     }];

//     let arguments = (ins AnyClassical:$val);

//     let assemblyFormat = [{
//         attr-dict $val `:` type($val)
//     }];
// }

def SYS_ParallelControlFlowOp : SYS_Op<"parallel_control_flow", [
                        SingleBlockImplicitTerminator<"ParallelEndOp">,
                        RecursiveSideEffects]> {
    let summary = "Contain a group of control flow ops that can run in parallel";
    let description = [{
        The `sys.parallel_control_flow` operation has a single region and block
        that holds a collection of scf control flow ops that can be executed in
        parallel.

        Example:
        ```mlir
        sys.parallel_control_flow {
            scf.if (%c0) {
                quir.gate_call @x(%q0) : (!quir.qubit<1>) -> ()
            } {quir.physicalIds = [0 : i32]}
            scf.if (%c1) {
                quir.gate_call @x(%q1) : (!quir.qubit<1>) -> ()
            } {quir.physicalIds = [1 : i32]}
        }
        ```
    }];

    let regions = (region SizedRegion<1>:$region);

    let assemblyFormat = "$region attr-dict";
}

def SYS_ParallelEndOp : SYS_Op<"parallel_control_flow_end", [
    Terminator, HasParent<"ParallelControlFlowOp">, NoSideEffect]> {
  let summary = "A pseudo op that marks the end of a `sys.parallel_control_flow` op.";
  let description = [{
    This op terminates the only block inside the only region of a
    `sys.parallel_control_flow` op.
  }];

  let parser = "return success();";
  let printer = "p << getOperationName();";
}

#endif // SYSTEM_OPS
