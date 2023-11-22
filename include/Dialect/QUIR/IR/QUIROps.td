//===- QUIROps.td - QUIR dialect ops -----------------------*- tablegen -*-===//
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

#ifndef QUIR_OPS
#define QUIR_OPS

include "Dialect/QUIR/IR/QUIRInterfaces.td"
include "Dialect/QUIR/IR/QUIRTraits.td"
include "Dialect/QUIR/IR/QUIRAttributes.td"

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

def QUIR_ConstantOp : QUIR_Op<"constant",
    [ConstantLike, Pure,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
     AllTypesMatch<["value", "result"]>]> {
  let summary = "QUIR-specific (angle or duration) constants";
  let description = [{
    The `quir.constant` operation produces an SSA value equal to some constant
    specified by an attribute. Created to support QUIR-specific constants, but
    can support any buildable attribute with a matching result type.

    Example:

    ```
    // Angle constant
    %1 = quir.constant #quir.angle<1.5 : !quir.angle>
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);


  let extraClassDeclaration = [{
    /// Whether the constant op can be constructed with a particular value and
    /// type.
    static bool isBuildableWith(Attribute value, Type type);

    // Return the angle value from the value attribute
    APFloat getAngleValueFromConstant();

    /// Build the constant op with `value` and `type` if possible, otherwise
    /// returns null.
    static ConstantOp materialize(OpBuilder &builder, Attribute value,
                                  Type type, Location loc);
  }];

  let hasFolder = 1;
  let assemblyFormat = "attr-dict $value";
}

def QUIR_DeclareQubitOp : QUIR_Op<"declare_qubit", [NonInterferingNonDeadSideEffect]> {
    let summary = "Declare a new physical qubit.";
    let description = [{
        The `quir.declare_qubit` operation creates a new physical qubit.
        Example:

        ```mlir
        %1 = quir.declare_qubit {id = 0 : i32}: !quir.qubit<1>
        ```
    }];

    let arguments = (ins OptionalAttr<I32Attr>:$id);
    let results = (outs AnyQubit:$res);

    let assemblyFormat = [{
        attr-dict `:` type($res)
    }];
}

def QUIR_ResetQubitOp : QUIR_Op<"reset", [
            IsolatedFromAbove, CPTPOp,
            DeclareOpInterfaceMethods<QubitOpInterface, ["getOperatedQubits"]>,
            NonInterferingNonDeadSideEffect]> {
    let summary = "Perform a reset operation on the given qubits";
    let description = [{
        The `quir.reset` operation performs a reset operation on all the qubits
        given as an argument. If multiple qubits are given, then the reset
        operations may happen in parallel.

        Example
        ```mlir
        quir.reset %qc0 : !quir.qubit<1>
        quir.reset %qc1, %qc2 : !quir.qubit<1>, !quir.qubit<1>
        ```
    }];

    let arguments = (ins Variadic<Qubit<1>>:$qubits);

    let assemblyFormat = [{
        attr-dict $qubits `:` type($qubits)
    }];
}

def QUIR_BuiltinCXOp : QUIR_Op<"builtin_CX", [
            UnitaryOp,
            DeclareOpInterfaceMethods<QubitOpInterface, ["getOperatedQubits"]>,
            NonInterferingNonDeadSideEffect]> {
    let summary = "Controlled NOT gate";
    let description = [{
        The `quir.builtin_CX` operation performs a controlled not i.e. flips %target iff %control is a 1.
        Example:
        ```mlir
        quir.builtin_CX %control, %target : !quir.qubit<1>, !quir.qubit<1>
        ```
    }];

    let arguments = (ins AnyQubit:$control, AnyQubit:$target);

    let assemblyFormat = [{
        attr-dict $control `,` $target `:` type($control) `,` type($target)
    }];
}

def QUIR_Builtin_UOp : QUIR_Op<"builtin_U", [
            UnitaryOp,
            DeclareOpInterfaceMethods<QubitOpInterface, ["getOperatedQubits"]>,
            NonInterferingNonDeadSideEffect]> {
    let summary = "Generic unitary gate";
    let description = [{
        The `quir.builtin_U` operation performs a single qubit unitary rotation.
        Example:
        ```mlir
        quir.builtin_U %target, %theta_0, %phi_0, %lambda_0 : !quir.qubit<1>, !quir.angle<1>, !quir.angle<1>, !quir.angle<1>
        ```
    }];

    let arguments = (ins AnyQubit:$target, AnyAngle:$theta, AnyAngle:$phi, AnyAngle:$lambda);

    let assemblyFormat = [{
        attr-dict $target `,` $theta `,` $phi `,` $lambda `:` type($target) `,` type($theta) `,` type($phi) `,` type($lambda)
    }];
}

def QUIR_CallGateOp : QUIR_Op<"call_gate", [CallOpInterface,
        UnitaryOp,
        DeclareOpInterfaceMethods<QubitOpInterface, ["getOperatedQubits"]>,
        NonInterferingNonDeadSideEffect]> {
    let summary = "A call to a user-defined quantum gate that needs to be specialized for input operands";
    let description = [{
        The `quir.call_gate` operation represents calls to user-defined quantum gates
        that needs to be specialized for its arguments. The callee gate is attached to a symbol reference via
        an attribute. The arguments list must match the arguments provided by the callee. For example:

        ```mlir
        quir.call_gate @userGateZX(%target, %theta, %phi) : (!quir.qubit<1>, !quir.angle<1>, !quir.angle<1>) -> ()
        ```
        This is valid only if the named user gate `userGateZX` exists and takes a qubit and 2 angles as arguments.
    }];

    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyAngleOrQubit>:$operands);
    let results = (outs );

    let assemblyFormat = [{
        $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
    }];

    let builders = [
        OpBuilder<(ins "mlir::func::FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", SymbolRefAttr::get(callee));
        }]>,
        OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$res,
        CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", callee);
        }]>,
        OpBuilder<(ins "StringRef":$callee, "TypeRange":$res,
        CArg<"ValueRange", "{}">:$operands), [{
            build($_builder, $_state, SymbolRefAttr::get($_builder.getContext(), callee), operands);
        }]>];

    let extraClassDeclaration = [{
        FunctionType getCalleeType();

        /// Get the argument operands to the called function.
        operand_range getArgOperands() {
            return {arg_operand_begin(), arg_operand_end()};
        }

        MutableOperandRange getArgOperandsMutable() {
            return getOperandsMutable();
        }

        operand_iterator arg_operand_begin() { return operand_begin(); }
        operand_iterator arg_operand_end() { return operand_end(); }

        /// Return the callee of this operation.
        CallInterfaceCallable getCallableForCallee() {
            return (*this)->getAttrOfType<SymbolRefAttr>("callee");
        }

        /// Set the callee for this operation.
        void setCalleeFromCallable(CallInterfaceCallable callee) {
            (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
        }

    }];
}

def QUIR_CallDefCalGateOp : QUIR_Op<"call_defcal_gate", [
            CallOpInterface,
            UnitaryOp,
            NonInterferingNonDeadSideEffect]> {
    let summary = "A call to a user-defined defcal quantum gate that needs to be specialized for input operands";
    let description = [{
        The `quir.call_defcal_gate` operation represents calls to user-defined quantum gates controlled with pulse-level descriptions
        that needs to be specialized for its arguments. The callee gate is attached to a symbol reference via
        an attribute. The arguments list must match the arguments provided by the callee. For example:

        ```mlir
        quir.call_defcal_gate @defcalGateZX(%target, %theta, %phi) : (!quir.qubit<1>, !quir.angle<1>, !quir.angle<1>) ->
        ```
        This is valid only if the named user gate `defcalGateZX` exists and takes a qubit and 2 angles as arguments.
    }];

    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyAngleOrQubit>:$operands);
    let results = (outs );

    let assemblyFormat = [{
        $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
    }];

    let builders = [
        OpBuilder<(ins "mlir::func::FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", SymbolRefAttr::get(callee));
        }]>,
        OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$res,
        CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", callee);
        }]>,
        OpBuilder<(ins "StringRef":$callee, "TypeRange":$res,
        CArg<"ValueRange", "{}">:$operands), [{
            build($_builder, $_state, SymbolRefAttr::get($_builder.getContext(), callee), operands);
        }]>];

    let extraClassDeclaration = [{
        /// Get the argument operands to the called function.
        operand_range getArgOperands() {
            return {arg_operand_begin(), arg_operand_end()};
        }

        operand_iterator arg_operand_begin() { return operand_begin(); }
        operand_iterator arg_operand_end() { return operand_end(); }

        /// Return the callee of this operation.
        CallInterfaceCallable getCallableForCallee() {
            return (*this)->getAttrOfType<SymbolRefAttr>("callee");
        }

        /// Set the callee for this operation.
        void setCalleeFromCallable(CallInterfaceCallable callee) {
            (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
        }

    }];
}

def QUIR_CallDefcalMeasureOp : QUIR_Op<"call_defcal_measure", [
            CallOpInterface,
            CPTPOp,
            NonInterferingNonDeadSideEffect]> {
    let summary = "A call to a user-defined defcal qubit measurement that needs to be specialized for input operands";
    let description = [{
        The `quir.call_defcal_measure` operation represents calls to user-defined qubit measurements controlled with pulse-level descriptions
        that needs to be specialized for its arguments. The callee gate is attached to a symbol reference via
        an attribute. The arguments list must match the arguments provided by the callee. For example:

        ```mlir
        %c1 = quir.call_defcal_measure @defcalMeasure(%target) : (!quir.qubit<1>) -> i1
        ```
    }];

    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyAngleOrQubit>:$operands);
    let results = (outs I1:$res);

    let assemblyFormat = [{
        $callee `(` $operands `)` attr-dict `:` functional-type(operands, results)
    }];

    let builders = [
        OpBuilder<(ins "mlir::func::FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", SymbolRefAttr::get(callee));
            $_state.addTypes(callee.getFunctionType().getResults());
        }]>,
        OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$res,
        CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", callee);
            $_state.addTypes(res);
        }]>,
        OpBuilder<(ins "StringRef":$callee, "TypeRange":$res,
        CArg<"ValueRange", "{}">:$operands), [{
            build($_builder, $_state, SymbolRefAttr::get($_builder.getContext(), callee), res,
                    operands);
        }]>];

    let extraClassDeclaration = [{
        /// Get the argument operands to the called function.
        operand_range getArgOperands() {
            return {arg_operand_begin(), arg_operand_end()};
        }

        operand_iterator arg_operand_begin() { return operand_begin(); }
        operand_iterator arg_operand_end() { return operand_end(); }

        /// Return the callee of this operation.
        CallInterfaceCallable getCallableForCallee() {
            return (*this)->getAttrOfType<SymbolRefAttr>("callee");
        }

        /// Set the callee for this operation.
        void setCalleeFromCallable(CallInterfaceCallable callee) {
            (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
        }

    }];

    let hasVerifier = 1;
}

def QUIR_CallSubroutineOp : QUIR_Op<"call_subroutine", [CallOpInterface]> {
    let summary = "A call to a user-defined subroutine function that needs to be specialized for input operands";
    let description = [{
        The `quir.call_subroutine` operation represents calls to user-defined subroutine functions
        that needs to be specialized for its arguments. The callee function is attached to a symbol reference via
        an attribute. The arguments list must match the argument types (but not widths) provided by the callee.
        Subroutines may receive and return classical types and may receive qubits as arguments.

        Example:
        ```mlir
        %majority = quir.call_subroutine @phase_4times(%q1, %phi) : (!quir.qubit<1>, !quir.angle<20>) -> ()
        ```

        Function localization and specialization can be performed to enable this call to be transformed into
        a std.call op.
    }];

    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyTypeOf<[AnyClassical, AnyQubit]>>:$operands);
    let results = (outs Optional<AnyClassical>:$res);

    let assemblyFormat = [{
        $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
    }];

    let builders = [
        OpBuilder<(ins "mlir::func::FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", SymbolRefAttr::get(callee));
            $_state.addTypes(callee.getFunctionType().getResults());
        }]>,
        OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$res,
        CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", callee);
            $_state.addTypes(res);
        }]>,
        OpBuilder<(ins "StringRef":$callee, "TypeRange":$res,
        CArg<"ValueRange", "{}">:$operands), [{
            build($_builder, $_state, SymbolRefAttr::get($_builder.getContext(), callee), res,
                    operands);
        }]>];

    let extraClassDeclaration = [{

        /// Get the argument operands to the called function.
        operand_range getArgOperands() {
            return {arg_operand_begin(), arg_operand_end()};
        }

        operand_iterator arg_operand_begin() { return operand_begin(); }
        operand_iterator arg_operand_end() { return operand_end(); }

        /// Return the callee of this operation.
        CallInterfaceCallable getCallableForCallee() {
            return (*this)->getAttrOfType<SymbolRefAttr>("callee");
        }

        /// Set the callee for this operation.
        void setCalleeFromCallable(CallInterfaceCallable callee) {
            (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
        }

    }];
}

def QUIR_CallCircuitOp : QUIR_Op<"call_circuit", [CallOpInterface, MemRefsNormalizable, DeclareOpInterfaceMethods<SymbolUserOpInterface>, DeclareOpInterfaceMethods<QubitOpInterface, ["getOperatedQubits"]>, NonInterferingNonDeadSideEffect]> {
    let summary = "Call a circuit operation";
    let description = [{
        The `quir.call_circuit` operation represents calls to launch a quantum circuit in the target system.
        The callee function is attached to a symbol reference via an attribute. The arguments list must match
        the argument types provided by the callee. The calling of a circuit represents the classical<->quantum
        interaction within the specified program. All classical input values to the circuit should be transferred
        before the invocation of the circuit routine to enable deterministic execution of the quantum circuit.

        TODO: Currently all qubits must be declared within the circuit for lowering to hardware due to how
        qubit allocations are currently tracked with `declare_qubit`. In the future we should consider
        supporting quantum arguments (qubits) to make truly reuseable circuit routines that align
        with the Qiskit and OpenQASM3 circuit definitions.

        Example:
        ```mlir
        %classical_result = quir.call_circuit @vqe(%theta) : (quir.angle<32>) -> (i1, i1)
        ```
    }];

    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyTypeOf<[AnyClassical, AnyQubit]>>:$operands);
    let results = (outs Variadic<AnyClassical>:$res);

    let assemblyFormat = [{
        $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
    }];

    let builders = [
        OpBuilder<(ins "CircuitOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", SymbolRefAttr::get(callee));
            $_state.addTypes(callee.getFunctionType().getResults());
        }]>,
        OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$res,
        CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", callee);
            $_state.addTypes(res);
        }]>,
        OpBuilder<(ins "StringRef":$callee, "TypeRange":$res,
        CArg<"ValueRange", "{}">:$operands), [{
            build($_builder, $_state, SymbolRefAttr::get($_builder.getContext(), callee), res,
                    operands);
        }]>];

    let extraClassDeclaration = [{
        FunctionType getCalleeType();

        /// Get the argument operands to the called function.
        operand_range getArgOperands() {
            return {arg_operand_begin(), arg_operand_end()};
        }

        MutableOperandRange getArgOperandsMutable() {
            return getOperandsMutable();
        }

        operand_iterator arg_operand_begin() { return operand_begin(); }
        operand_iterator arg_operand_end() { return operand_end(); }

        /// Return the callee of this operation.
        CallInterfaceCallable getCallableForCallee() {
            return (*this)->getAttrOfType<SymbolRefAttr>("callee");
        }

        /// Set the callee for this operation.
        void setCalleeFromCallable(CallInterfaceCallable callee) {
            (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
        }
    }];
}



def QUIR_MeasureOp : QUIR_Op<"measure", [
            CPTPOp,
            DeclareOpInterfaceMethods<QubitOpInterface, ["getOperatedQubits"]>,
            NonInterferingNonDeadSideEffect]> {
    let summary = "Measure qubits";
    let description = [{
        The `quir.measure` operation represents a quantum measurement
        performed in parallel on multiple qubits. It returns a number of
        classical bits equal to the number of input qubits.

        Example:
        ```mlir
        %c1, %c2 = quir.measure(%q1, %q2) : (!quir.qubit<1>, !quir.qubit<1>) -> (i1, i1)
        ```
    }];

    let arguments = (ins Variadic<Qubit<1>>:$qubits);
    let results = (outs Variadic<I1>:$outs);

    let assemblyFormat = [{
        `(` $qubits `)` attr-dict `:` functional-type($qubits, $outs)
    }];
}

def QUIR_DelayOp : QUIR_Op<"delay", [
            UnitaryOp,
            DeclareOpInterfaceMethods<QubitOpInterface, ["getOperatedQubits"]>,
            NonInterferingNonDeadSideEffect]> {
    let summary = "Add a delay operation of a given duration or stretch to a qubit";
    let description = [{
        The `quir.delay` operation represents a delay operation of the given
        duration or stretch to a qubit, group of qubits, or all qubits (when no
        target qubit is given).

        Example:
        ```mlir
        %dur1 = quir.constant #quir.duration<10.0 : !quir.duration<dt>>
        "quir.delay"(%dur1, %q1_1) : (!quir.duration<dt>, !quir.qubit)
        ```
    }];

    let arguments = (ins DurationOrStretch:$time, Variadic<AnyQubit>:$qubits);
    let results = (outs );

    let assemblyFormat = [{
        attr-dict $time `,` `(` $qubits `)` `:` type($time) `,` functional-type($qubits, results)
    }];
}

def QUIR_BarrierOp : QUIR_Op<"barrier", [
            UnitaryOp,
            DeclareOpInterfaceMethods<QubitOpInterface, ["getOperatedQubits"]>,
            NonInterferingNonDeadSideEffect]> {
    let summary = "Add a barrier operation";
    let description =   [{
        The `quir.barrier` operation represents a barrier operation that synchronizes subsequent
        operations across all qubits. It ensures that all subsequent operations are blocked until
        the last operation of the input set is complete.

        Example:
        ```mlir
        quir.barrier %qc0 : (!quir.qubit<1>) -> ()
        quir.barrier %qc0, %qc1 : (!quir.qubit<1>, !quir.qubit<1>) -> ()
        ```
    }];

    let arguments = (ins Variadic<AnyQubit>:$qubits);
    let results = (outs);

    let assemblyFormat = [{
        $qubits attr-dict `:` functional-type($qubits, results)
    }];
}

//===----------------------------------------------------------------------===//
//
// This code section was derived and modified from the LLVM project
// Consequently it is licensed as described below.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
def QUIR_CircuitOp : QUIR_Op<"circuit", [
  AutomaticAllocationScope, CallableOpInterface,
  FunctionOpInterface, IsolatedFromAbove, Symbol,
  DeclareOpInterfaceMethods<QubitOpInterface>,
  RecursiveMemoryEffects
]> {
  let summary = "An operation with a name containing a single `SSACFG` region corresponding to a quantum circuit(block) execution";
  let description = [{

    Defines a circuit(function) operation representing a
    proecedural (deterministically timed) circuit execution
    on the QPU *without* any classical compute or control-flow.
    Effectively corresponding to a callable "circuit block".

    The CircuitOp is modelled after the FuncOp of the MLIR func
    dialect.

    A circuit may only contain operations declaring qubits
    TODO: move qubit declaration/allocation outside of the circuit),
    unitary gates operations, and projective measurements. It may only
    be launched from the QPU's control-system directly.

    All non-constant classical values required to define the circuit must
    be provided as circuit arguments to allow a clean separation of
    classical and quantum processes within the MLIR.

    Quantum programs should be canonicalized such that all quantum
    operations occurr within a region contained by a `quir.circuit` operation
    and are reached through a corresponding `quir.call_circuit`.

    TODO: Clarify the difference between circuit and gate definitions.

    Operations within the circuit cannot implicitly capture values defined
    outside of the circuit, i.e. they are `IsolatedFromAbove`. All
    external references must use circuit arguments or attributes that establish
    a symbolic connection (e.g. symbols referenced by name via a string
    attribute like SymbolRefAttr). An external circuit declaration (used when
    referring to a circuit declared in some other module) has no body. While
    the MLIR textual form provides a nice inline syntax for circuit arguments,
    they are internally represented as “block arguments” to the first block in
    the region.

    Syntax:

    ```
    op ::= `quir.circuit` symbol-ref-id `(` argument-list `)` (`->`
    function-result-list)? function-attributes? region
    ```

    Example:

    ```mlir
    // External circuit function definitions.
    quir.circuit @x()

    // A function that returns its argument twice:
    quir.circuit @vqe(%angle: quir.angle<32>) -> i1
      {quir.physicalId = 0 : i32} {
      %q0 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
      quir.gate_call @rz(%0, %angle) : (!quir.qubit<1>, !quir.angle<32>) -> ()
      %result = "quir.measure"(%0) : (!quir.qubit<1>) -> i1
      quir.return %result: i1
    }
    ```
  }];

  let arguments = (ins SymbolNameAttr:$sym_name,
                       TypeAttrOf<FunctionType>:$function_type,
                       OptionalAttr<StrAttr>:$sym_visibility,
                       OptionalAttr<DictArrayAttr>:$arg_attrs,
                       OptionalAttr<DictArrayAttr>:$res_attrs);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
    >];
  let extraClassDeclaration = [{
    static CircuitOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs = {});
    static CircuitOp create(Location location, StringRef name, FunctionType type,
                         Operation::dialect_attr_range attrs);
    static CircuitOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs,
                         ArrayRef<DictionaryAttr> argAttrs);

    /// Create a deep copy of this circuit and all of its blocks, remapping any
    /// operands that use values outside of the circuit using the map that is
    /// provided (leaving them alone if no entry is present). If the mapper
    /// contains entries for circuit arguments, these arguments are not
    /// included in the new circuit. Replaces references to cloned sub-values
    /// with the corresponding value that is copied, and adds those mappings to
    /// the mapper.
    CircuitOp clone(IRMapping &mapper);
    CircuitOp clone();

    /// Clone the internal blocks and attributes from this circuit into dest.
    /// Any cloned blocks are appended to the back of dest. This circuit
    /// asserts that the attributes of the current circuit and dest are
    /// compatible.
    void cloneInto(CircuitOp dest, IRMapping &mapper);

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> getCallableResults() { return getFunctionType().getResults(); }

    /// Returns the argument attributes for all callable region arguments or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableArgAttrs() {
      return getArgAttrs().value_or(nullptr);
    }

    /// Returns the result attributes for all callable region results or
    /// null if there are none.
    ::mlir::ArrayAttr getCallableResAttrs() {
      return getResAttrs().value_or(nullptr);
    }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def QUIR_ReturnOp : QUIR_Op<"return", [HasParent<"CircuitOp">, Pure, Terminator, ReturnLike]>,
    Arguments<(ins Variadic<AnyClassical>:$operands)>, Results<(outs)> {
        let summary = "Terminator for quir.circuit.";
        let description = [{
                A terminator for regions that appear in the body of the `quir.circuit`
                circuit operation. The operands to the `quir.return` are the result values
                that are returned to the `quir.call_circuit` invocation of the target circuit.

                Example:

                ```mlir
                quir.circuit @foo(%angle: quir.angle<32>) -> (i1, i1)
                    ...
                    quir.return %0, %1: i1, i1
                }
            }];
        let builders = [OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>];
        let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
        let hasVerifier = 1;
}

def QUIR_SwitchOp : QUIR_Op<"switch",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
     RecursiveMemoryEffects,
     SingleBlockImplicitTerminator<"quir::YieldOp">,
     NoRegionArguments
     ]> {
    let summary = "switch to case regions based on the operand value";
    let description = [{
        The "quir.switch" operation conditionally switch to a region of code
        based on the operand value. The operand is of unsigned 32-integer.
        There is a default region right after the operand. Then, there are
        some case regions wrapped by square brackets. For example,

        ```mlir
        quir.switch %flag {
            ...
        } [
            0: {
                ...
            }
            1: {
                ...
            }
            ...
        ]
        ```

        "quir.switch" can also return results. In this case, each region should
        end with a "quir.yield" and the number and type of returned results
        should accord to what specified before the default region, e.g.,

        ```mlir
        %y = quir.switch %flag -> (i32) {
            %y_def = ...
            quir.yield %y_def : i32
        } [
            4: {
                %y_4 = ...
                quir.yield %y_4 : i32
            }
            ...
        ]
        ```
    }];
    let arguments = (ins I32:$flag,
                    I32ElementsAttr:$caseValues);
    let results = (outs Variadic<AnyType>:$resultTypes);
    let regions = (region SizedRegion<1>:$defaultRegion,
                    VariadicRegion<SizedRegion<1>>:$caseRegions);

    let hasVerifier = 1;
    let hasCustomAssemblyFormat = 1;
}

def YieldOp : QUIR_Op<"yield", [Pure, ReturnLike, Terminator, ParentOneOf<["SwitchOp"]>]>{
    let summary = "yield and termination operation";
    let description = [{
        Duplication of "scf.yield" at https://github.com/llvm/llvm-project/blob
        /release/12.x/mlir/include/mlir/Dialect/SCF/SCFOps.td
        Also duplication of "affine.yield" at https://github.com/llvm/llvm-project
        /blob/release/12.x/mlir/include/mlir/Dialect/Affine/IR/AffineOps.td
        "quir.yield" yields an SSA value from the QUIR dialect op region and
        terminates the regions, e.g., the case regions in "quir.switch".
        The semantics of how the values are yielded is defined by the parent
        operation. We also need to specify the parent op in the traits of yield.
        If "quir.yield" has any operands, the operands must match the parent
        operation's results.
        If the parent operation defines no values, then the "quir.yield" may be
        left out in the custom syntax and the builders will insert one implicitly.
        Otherwise, it has to be present in the syntax to indicate which values are
        yielded.
    }];

    let arguments = (ins Variadic<AnyType>:$results);
    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
    let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
    let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// end Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//



#endif // QUIR_OPS
