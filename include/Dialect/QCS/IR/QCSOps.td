//===- QCSOps.td - System dialect ops ----------------------*- tablegen -*-===//
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
/// This is the main operation definition file for Quantum Control System
/// operations.
///
//===----------------------------------------------------------------------===//

#ifndef QCS_OPS
#define QCS_OPS

// TODO: Temporary, until constraints between `OpenQASM3`, `QUIR`, `Pulse`, and
// `System` dialects are ironed out.
include "Dialect/QUIR/IR/QUIRInterfaces.td"
include "Dialect/QUIR/IR/QUIRTraits.td"
include "Dialect/QUIR/IR/QUIRTypeConstraints.td"

include "Dialect/QCS/IR/QCSBase.td"

include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/SymbolInterfaces.td"

// Define a side effect that identifies an operation as not dead while not
// interfering with memory operations (e.g., allows store-forwarding across
// this operation).
// Note that operations without memory effects defined will be treated
// conservatively (i.e., not making any assumptions).
// see lib/Interfaces/SideEffectInterfaces.cpp:isTriviallyDeadImpl()
// see lib/Dialect/Affine/Utils/Utils.cpp:hasNoInterveningEffect()
def NonInterferingNonDeadSideEffect : MemoryEffects<[MemFree<DefaultResource>]>;

def QCS_SystemInitOp : QCS_Op<"init", [IsolatedFromAbove]> {
    let summary = "Initializes the system";
    let description = [{
        The `qcs.system_init` operation causes the system to be initialized,
        preparing for execution and synchronizing the system.
    }];
    let assemblyFormat = [{
        attr-dict
    }];
}

def QCS_SystemFinalizeOp : QCS_Op<"finalize", [IsolatedFromAbove]> {
    let summary = "Finalizes the system";
    let description = [{
        The `qcs.system_finalize` operation causes the system to be finalized,
        cleaning up results and shutting down execution.
    }];
    let assemblyFormat = [{
        attr-dict
    }];
}

def QCS_SynchronizeOp : QCS_Op<"synchronize", [
            DeclareOpInterfaceMethods<QubitOpInterface, ["getOperatedQubits"]>,
            NonInterferingNonDeadSideEffect]> {
    let summary = "Synchronize the system";
    let description = [{
        The `qcs.synchronize` operation causes the system to perform a
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

def QCS_BroadcastOp : QCS_Op<"broadcast", [NonInterferingNonDeadSideEffect]> {
    let summary = "Broadcast a value from this controller to all others";
    let description = [{
        The `qcs.broadcast` operation represents a broadcast command that sends a value
        from this controller to all others. All other controllers should have a
        corresponding `qcs.recv` operation. If only one controller should receive the
        message then the `qcs.send` operation should be used instead.

        Example:
        ```mlir
        %angle1 = quir.constant #quir.angle<0.2 : !quir.angle<20>>
        qcs.broadcast %angle1 : !quir.angle<20>
        ```
    }];

    let arguments = (ins AnyClassical:$val);

    let assemblyFormat = [{
        attr-dict $val `:` type($val)
    }];
}

// TODO: Should this be a "unitary op" at the system level?
def QCS_DelayCyclesOp : QCS_Op<"delay_cycles", [
            UnitaryOp,
            DeclareOpInterfaceMethods<QubitOpInterface, ["getOperatedQubits"]>,
            NonInterferingNonDeadSideEffect]> {
    let summary = "Add a delay operation of a given number of cycles to a qubit";
    let description = [{
        The `qcs.delay_cycles` operation represents a delay operation of the given
        number of cycles to a qubit, group of qubits, or all qubits (when no
        target qubit is given).

        Example:
        ```mlir
        "qcs.delay_cycles"(%q1_1) { time = 1000 : i64}: (!quir.qubit) -> ()
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

def QCS_SendOp : QCS_Op<"send", [NonInterferingNonDeadSideEffect]> {
    let summary = "Send a classical value from this controller to another";
    let description = [{
        The `qcs.send` operation represents a send command from one controller to another.
        A corresponding `qcs.recv` operation should receive the information sent.

        Example:
        ```mlir
        %cbit = "quir.measure"(%target) : (!quir.qubit<1>) -> i1
        qcs.send %cbit : i1
        ```
    }];

    let arguments = (ins AnyClassical:$val, IndexAttr:$id);

    let assemblyFormat = [{
        attr-dict $val `to` $id `:` type($val)
    }];
}

def QCS_RecvOp : QCS_Op<"recv", [NonInterferingNonDeadSideEffect]> {
    let summary = "Receive classical values from other controllers";
    let description = [{
        The `qcs.recv` operation represents a receive command for potentially
        multiple values. The fromIds array attribute indicates the Id of the
        sending control node for each independent value.

        Example:
        ```mlir
        %meas:2 = qcs.recv {fromId = [0 : index, 1 : index]} : i1, i1
        ```
    }];

    let arguments = (ins OptionalAttr<IndexArrayAttr>:$fromIds);
    let results = (outs Variadic<AnyClassical>:$vals);

    let assemblyFormat = [{
        attr-dict `:` type($vals)
    }];
}

def QCS_ParallelControlFlowOp : QCS_Op<"parallel_control_flow", [
                        SingleBlockImplicitTerminator<"ParallelEndOp">,
                        RecursiveSideEffects]> {
    let summary = "Contain a group of control flow ops that can run in parallel";
    let description = [{
        The `qcs.parallel_control_flow` operation has a single region and block
        that holds a collection of scf control flow ops that can be executed in
        parallel.

        Example:
        ```mlir
        qcs.parallel_control_flow {
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

def QCS_ParallelEndOp : QCS_Op<"parallel_control_flow_end", [
    Terminator, HasParent<"ParallelControlFlowOp">, NoSideEffect]> {
  let summary = "A pseudo-op that marks the end of a `qcs.parallel_control_flow` op.";
  let description = [{
    This op terminates the only block inside the only region of a
    `qcs.parallel_control_flow` op.
  }];

  let parser = "return success();";
  let printer = "p << getOperationName();";
}

def QCS_ShotInitOp : QCS_Op<"shot_init", [IsolatedFromAbove]> {
    let summary = "Initialization for shots";
    let description = [{
        The `qcs.shot_init` operation causes the system to initialize a shot to ensure
        consistent initial conditions for each execution of a given quantum algorithm.
    }];
    let assemblyFormat = [{
        attr-dict
    }];
}

def QCS_DeclareParameterOp : QCS_Op<"declare_parameter", [Symbol]> {
    let summary = "system input parameter subject to post compilation binary rewrite";
    let description = [{
        The `qcs.declare_parameter` operation adds a symbol defining an input parameter
        which may be modified after compilation. The value of the input parameter
        may be obtained using the qcs.use_input_parameter operation.

        Example:

        ```
        // quir.angle input parameter
        qcs.declare_parameter  "theta" : !quir.angle<64> = 3.14159
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type,
        OptionalAttr<AnyAttr>:$initial_value
    );

    let results = (outs);

    let assemblyFormat = [{
        attr-dict $sym_name `:` $type (`=` $initial_value^)?
    }];

    let builders = [
        OpBuilder<(ins "::llvm::StringRef":$sym_name, "::mlir::TypeAttr":$type), [{
            $_state.addAttribute("sym_name", $_builder.getStringAttr(sym_name));
            $_state.addAttribute("type", type);
        }]>,
        OpBuilder<(ins "::llvm::StringRef":$sym_name, "::mlir::TypeAttr":$type, "Attribute":$value), [{
            $_state.addAttribute("sym_name", $_builder.getStringAttr(sym_name));
            $_state.addAttribute("type", type);
            $_state.addAttribute("initial_value", value);
        }]>,
    ];

    let verifier = [{
        auto t = (*this).type();

        if( t.isa<::mlir::quir::AngleType>() )
            return success();
        std::string str;
        llvm::raw_string_ostream os(str);
        t.print(os);

        return emitOpError("MLIR type " + str + " not supported for declarations.");
    }];
}

def QCS_ParameterLoadOp : QCS_Op<"parameter_load",
                        [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Use the current value of a parameter";
    let description = [{
        The operation `qcs.parameter_load` returns the current value of the
        classical parameter with the given name.

        Example:

        ```mlir
        %2 = qcs.parameter_load "a" : !quir.angle<64>
        ```
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$parameter_name
    );

    let results = (outs AnyClassical:$res);

    let assemblyFormat = [{
        $parameter_name `:` type($res) attr-dict
    }];
    // op is verified by its traits
    let verifier = ?;
}


#endif // QCS_OPS
