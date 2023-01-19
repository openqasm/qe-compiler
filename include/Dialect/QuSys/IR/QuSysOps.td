//===- SystemOps.td - System dialect ops -------------------*- tablegen -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This is the main operation definition file for Quantum System operations.
///
//===----------------------------------------------------------------------===//

#ifndef QUSYS_OPS
#define QUSYS_OPS

// TODO: Temporary, until constraints between `OpenQASM3`, `QUIR`, `Pulse`, and
// `System` dialects are ironed out.
include "Dialect/QUIR/IR/QUIRInterfaces.td"
include "Dialect/QUIR/IR/QUIRTraits.td"
include "Dialect/QUIR/IR/QUIRTypeConstraints.td"

include "Dialect/QuSys/IR/QuSysBase.td"

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"

// Define a side effect that identifies an operation as not dead while not
// interfering with memory operations (e.g., allows store-forwarding across
// this operation).
// Note that operations without memory effects defined will be treated
// conservatively (i.e., not making any assumptions).
// see lib/Interfaces/SideEffectInterfaces.cpp:isTriviallyDeadImpl()
// see lib/Dialect/Affine/Utils/Utils.cpp:hasNoInterveningEffect()
def NonInterferingNonDeadSideEffect : MemoryEffects<[MemFree<DefaultResource>]>;

def QuSys_SystemInitOp : QuSys_Op<"init", [IsolatedFromAbove]> {
    let summary = "Initializes the system";
    let description = [{
        The `qusys.system_init` operation causes the system to be initialized,
        preparing for execution and synchronizing the system.
    }];
    let assemblyFormat = [{
        attr-dict
    }];
}

def QuSys_SystemFinalizeOp : QuSys_Op<"finalize", [IsolatedFromAbove]> {
    let summary = "Finalizes the system";
    let description = [{
        The `qusys.system_finalize` operation causes the system to be finalized,
        cleaning up results and shutting down execution.
    }];
    let assemblyFormat = [{
        attr-dict
    }];
}

def QuSys_SynchronizeOp : QuSys_Op<"synchronize", [
            DeclareOpInterfaceMethods<QubitOpInterface, ["getOperatedQubits"]>,
            NonInterferingNonDeadSideEffect]> {
    let summary = "Synchronize the system";
    let description = [{
        The `qusys.synchronize` operation causes the system to perform a
        synchronization so that each of the independent controllers is locked
        into the same time step again. The synchronize op accepts qubit
        arguments, this is to simplify the tracking of synchronization timing
        relative to other qubit-based operations, like measurement and reset.
    }];
    let arguments = (ins Variadic<AnyQubit>:$qubits);
    let results = (outs);

    let assemblyFormat = [{
        $qubits attr-dict `:` functional-type($qubits, results)
    }];
}

def QuSys_BroadcastOp : QuSys_Op<"broadcast", [NonInterferingNonDeadSideEffect]> {
    let summary = "Broadcast a value from this controller to all others";
    let description = [{
        The `qusys.broadcast` operation represents a broadcast command that sends a value
        from this controller to all others. All other controllers should have a
        corresponding `qusys.recv` operation. If only one controller should receive the
        message then the `qusys.send` operation should be used instead.

        Example:
        ```mlir
        %angle1 = quir.constant #quir.angle<0.2 : !quir.angle<20>>
        qusys.broadcast %angle1 : !quir.angle<20>
        ```
    }];

    let arguments = (ins AnyClassical:$val);

    let assemblyFormat = [{
        attr-dict $val `:` type($val)
    }];
}

// TODO: Should this be a "unitary op" at the system level?
def QuSys_DelayCyclesOp : QuSys_Op<"delay_cycles", [
            UnitaryOp,
            DeclareOpInterfaceMethods<QubitOpInterface, ["getOperatedQubits"]>,
            NonInterferingNonDeadSideEffect]> {
    let summary = "Add a delay operation of a given number of cycles to a qubit";
    let description = [{
        The `qusys.delay_cycles` operation represents a delay operation of the given
        number of cycles to a qubit, group of qubits, or all qubits (when no
        target qubit is given).

        Example:
        ```mlir
        "qusys.delay_cycles"(%q1_1) { time = 1000 : i64}: (!quir.qubit) -> ()
        ```
    }];

    let arguments = (ins I64Attr:$time, Variadic<AnyQubit>:$qubits);
    let results = (outs );

    let assemblyFormat = [{
        `(` $qubits `)` attr-dict `:` functional-type($qubits, results)
    }];
}

def IndexArrayAttr : TypedArrayAttrBase<IndexAttr,
                                        "index array attribute"> {
  let constBuilderCall = "$_builder.getIndexArrayAttr($0)";
}

def QuSys_SendOp : QuSys_Op<"send", [NonInterferingNonDeadSideEffect]> {
    let summary = "Send a classical value from this controller to another";
    let description = [{
        The `qusys.send` operation represents a send command from one controller to another.
        A corresponding `qusys.recv` operation should receive the information sent.

        Example:
        ```mlir
        %cbit = "quir.measure"(%target) : (!quir.qubit<1>) -> i1
        qusys.send %cbit : i1
        ```
    }];

    let arguments = (ins AnyClassical:$val, IndexAttr:$id);

    let assemblyFormat = [{
        attr-dict $val `to` $id `:` type($val)
    }];
}

def QuSys_RecvOp : QuSys_Op<"recv", [NonInterferingNonDeadSideEffect]> {
    let summary = "Receive classical values from other controllers";
    let description = [{
        The `qusys.recv` operation represents a receive command for potentially
        multiple values. The fromIds array attribute indicates the Id of the
        sending control node for each independent value.

        Example:
        ```mlir
        %meas:2 = qusys.recv {fromId = [0 : index, 1 : index]} : i1, i1
        ```
    }];

    let arguments = (ins OptionalAttr<IndexArrayAttr>:$fromIds);
    let results = (outs Variadic<AnyClassical>:$vals);

    let assemblyFormat = [{
        attr-dict `:` type($vals)
    }];
}

def QuSys_ParallelControlFlowOp : QuSys_Op<"parallel_control_flow", [
                        SingleBlockImplicitTerminator<"ParallelEndOp">,
                        RecursiveSideEffects]> {
    let summary = "Contain a group of control flow ops that can run in parallel";
    let description = [{
        The `qusys.parallel_control_flow` operation has a single region and block
        that holds a collection of scf control flow ops that can be executed in
        parallel.

        Example:
        ```mlir
        qusys.parallel_control_flow {
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

def QuSys_ParallelEndOp : QuSys_Op<"parallel_control_flow_end", [
    Terminator, HasParent<"ParallelControlFlowOp">, NoSideEffect]> {
  let summary = "A pseudo-op that marks the end of a `qusys.parallel_control_flow` op.";
  let description = [{
    This op terminates the only block inside the only region of a
    `qusys.parallel_control_flow` op.
  }];

  let parser = "return success();";
  let printer = "p << getOperationName();";
}

def QuSys_ShotInitOp : QuSys_Op<"shot_init", [IsolatedFromAbove]> {
    let summary = "Initialization for shots";
    let description = [{
        The `qusys.shot_init` operation causes the system to initialize a shot to ensure
        consistent initial conditions for each execution of a given quantum algorithm.
    }];
    let assemblyFormat = [{
        attr-dict
    }];
}

#endif // QUSYS_OPS
