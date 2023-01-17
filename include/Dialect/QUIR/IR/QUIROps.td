//===- QUIROps.td - QUIR dialect ops -----------------------*- tablegen -*-===//
//
// (C) Copyright IBM 2021, 2022.
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

// TODO: This op should be extracted to a system-level dialect
def QUIR_ShotInitOp : QUIR_Op<"shot_init", [IsolatedFromAbove]> {
    let summary = "Initialization for shot loops";
    let description = [{
        The `quir.shot_init` operation causes the system to initialize the shot
        loop to maintain consistent initial conditions for each shot
    }];
    let assemblyFormat = [{
        attr-dict
    }];
}

// TODO: This op should be extracted to a system-level dialect
def QUIR_SynchronizeOp : QUIR_Op<"synchronize", [
            DeclareOpInterfaceMethods<QubitOpInterface, ["getOperatedQubits"]>,
            NonInterferingNonDeadSideEffect]> {
    let summary = "Synchronize the system";
    let description = [{
        The `quir.synchronize` operation causes the system to perform a
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

def QUIR_ConstantOp : QUIR_Op<"constant",
    [ConstantLike, NoSideEffect,
     DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
     TypesMatchWith<
    "result and attribute have the same type",
    "value", "result", "$_self">]> {
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

  let arguments = (ins AnyAttr:$value);
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder<(ins "Attribute":$value),
    [{ build($_builder, $_state, value.getType(), value); }]>,
    OpBuilder<(ins "Attribute":$value, "Type":$type),
    [{ build($_builder, $_state, type, value); }]>,
  ];

  let extraClassDeclaration = [{
    /// Whether the constant op can be constructed with a particular value and
    /// type.
    static bool isBuildableWith(Attribute value, Type type);

    // Return the angle value from the value attribute
    APFloat getAngleValueFromConstant();
  }];

  let hasFolder = 1;
  let assemblyFormat = "attr-dict $value";
}

def QUIR_DeclareVariableOp : QUIR_Op<"declare_variable", [Symbol]> {
    let summary = "Declares a classical variable";
    let description = [{
        The `quir.declare_variable` operation declares a classical variable
        with the given name (sym_name), type, and an optional constant
        initializer.
        If present, the attributes `input` and `output` indicate that this
        variable is an input or output variable, respectively.

        Example:

        ```mlir
        quir.declare_variable "myVar" : i1 = true
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type,
        UnitAttr:$input,
        UnitAttr:$output,
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
        }]>
    ];

    let verifier = [{
        auto t = (*this).type();

        if( t.isa<AngleType>() || t.isa<CBitType>() || t.isa<DurationType>() || t.isa<StretchType>() || t.isIntOrIndexOrFloat() || t.isa<ComplexType>())
            return success();
        std::string str;
        llvm::raw_string_ostream os(str);
        t.print(os);

        return emitOpError("MLIR type " + str + " not supported for declarations.");
    }];

    let extraClassDeclaration = [{
        bool isInputVariable() { return input(); }
        bool isOutputVariable() { return output(); }
    }];
}

def QUIR_DeclareArrayOp : QUIR_Op<"declare_array", [Symbol]> {
    let summary = "Declares a classical array";
    let description = [{
        The `quir.declare_array` operation declares a classical array
        with the given name (sym_name), element type, and number of elements.

        Example:
        ```mlir
        quir.declare_array "myVar" : i1[2]
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type,
        IndexAttr:$num_elements
    );

    let results = (outs);

    let assemblyFormat = [{
        attr-dict $sym_name `:` $type `[` $num_elements `]`
    }];

    let verifier = [{
        auto t = (*this).type();

        if( t.isa<AngleType>() || t.isa<CBitType>() || t.isa<DurationType>() || t.isa<StretchType>() || t.isIntOrIndexOrFloat())
            return success();
        return failure();
    }];
}

def QUIR_UseVariableOp : QUIR_Op<"use_variable",
                        [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Use the current value of a variable";
    let description = [{
        The operation `quir.use_variable` returns the current value (wrt to
        `quir.assign_variable` operations) of the classical variable with the
        given name.

        Example:

        ```mlir
        %2 = quir.use_variable "a" : !quir.cbit<1>
        ```
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$variable_name
    );

    let results = (outs AnyClassical:$res);

    let assemblyFormat = [{
        $variable_name `:` type($res) attr-dict
    }];
    // op is verified by its traits
    let verifier = ?;
}

def QUIR_UseArrayElementOp : QUIR_Op<"use_array_element",
                        [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Use the current value of an array element";
    let description = [{
        The operation `quir.use_array_element` returns the current value (wrt
        to `quir.assign_array_element` operations) of an element in a classical
        array, where the array is specified by name and the element by index.

        Example:

        ```mlir
        %2 = quir.use_array_element "result"[6] : i1
        ```
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$variable_name,
        IndexAttr:$index
    );

    let results = (outs AnyClassical:$res);

    let assemblyFormat = [{
        $variable_name `[` $index `]` `:` type($res) attr-dict
    }];
    // op is verified by its traits
    let verifier = ?;
}

def QUIR_VariableAssignOp : QUIR_Op<"assign_variable",
                        [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Assign a new value to a classical variable";
    let description = [{
        The operation `quir.assign_variable` assigns a new value to a classical
        variable given by name.
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$variable_name,
        AnyClassical:$assigned_value
    );

    let results = (outs);

    let assemblyFormat = [{
        attr-dict $variable_name `:` type($assigned_value) `=` $assigned_value
    }];
    // op is verified by its traits
    let verifier = ?;
}

def QUIR_AssignArrayElementOp : QUIR_Op<"assign_array_element",
                        [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Assigns a new value to an element of a classical array";
    let description = [{
        The operation `quir.assign_array_element` assigns a new value to an
        element of a classical array, where the array is specified by name and
        the element by index.
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$variable_name,
        IndexAttr:$index,
        AnyClassical:$assigned_value
    );

    let results = (outs);

    let assemblyFormat = [{
        attr-dict $variable_name `[` $index `]` `:` type($assigned_value) `=` $assigned_value
    }];
    // op is verified by its traits
    let verifier = ?;
}

def QUIR_AssignCbitBitOp : QUIR_Op<"assign_cbit_bit",
                        [DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
    let summary = "Assign a single bit in a classical bit register";
    let description = [{
        The operation `quir.assign_cbit_bit` assigns a new value to an
        individual bit of a classical bit register in a variable.

        Example:

        ```mlir
        quir.assign_cbit_bit "b"<2> [0] : i1 = %9
        ```
    }];

    let arguments = (ins
        FlatSymbolRefAttr:$variable_name,
        IndexAttr:$index,
        IndexAttr:$cbit_width,
        I1:$assigned_bit
    );

    let results = (outs);

    let assemblyFormat = [{
        attr-dict $variable_name `<` $cbit_width `>` `[` $index `]` `:` type($assigned_bit) `=` $assigned_bit
    }];
    // op is verified by its traits
    let verifier = ?;
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

def QUIR_DeclareDurationOp : QUIR_Op<"declare_duration", [NoSideEffect]> {
    let summary = "Declare a new duration.";
    let description = [{
        The `quir.declare_duration` operation creates a new OpenQASM3.0 duration, representing a duration of time.

        Example:
        ```mlir
        %dur = quir.declare_duration {value = "10ns"} : !quir.duration
        ```
    }];

    let arguments = (ins StrAttr:$value);
    let results = (outs QUIR_DurationType:$out);

    let assemblyFormat = [{
        attr-dict `:` type($out)
    }];
}

def QUIR_DeclareStretchOp : QUIR_Op<"declare_stretch", [NoSideEffect]> {
    let summary = "Declare a new stretch.";
    let description = [{
        The `quir.declare_stretch` operation creates a new OpenQASM3.0 stretch, representing an unknown duration of time.

        Example:
        ```mlir
        %dur = "quir.declare_stretch"() : () -> !quir.stretch
        ```
    }];

    let results = (outs QUIR_StretchType:$out);

    let assemblyFormat = [{
        attr-dict `:` type($out)
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

def QUIR_CallGateOp : QUIR_Op<"call_gate", [DeclareOpInterfaceMethods<CallOpInterface>,
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
        OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
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
        StringRef getCallee() { return callee(); }
        FunctionType getCalleeType();

        operand_iterator arg_operand_begin() { return operand_begin(); }
        operand_iterator arg_operand_end() { return operand_end(); }

    }];
}

def QUIR_CallDefCalGateOp : QUIR_Op<"call_defcal_gate", [
            DeclareOpInterfaceMethods<CallOpInterface>,
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
        OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
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
        StringRef getCallee() { return callee(); }
        FunctionType getCalleeType();

        operand_iterator arg_operand_begin() { return operand_begin(); }
        operand_iterator arg_operand_end() { return operand_end(); }

    }];
}

def QUIR_CallDefcalMeasureOp : QUIR_Op<"call_defcal_measure", [
            DeclareOpInterfaceMethods<CallOpInterface>,
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
        OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", SymbolRefAttr::get(callee));
            $_state.addTypes(callee.getType().getResults());
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
        StringRef getCallee() { return callee(); }
        FunctionType getCalleeType();

        operand_iterator arg_operand_begin() { return operand_begin(); }
        operand_iterator arg_operand_end() { return operand_end(); }

    }];

    let verifier = [{
        bool qubitFound = false;
        for (auto arg : (*this)->getOperands()) {
            if(qubitFound) {
                if(arg.getType().isa<QubitType>()) {
                    return emitOpError("requires exactly one qubit argument");
                }
            } else {
                if(arg.getType().isa<QubitType>()) {
                    qubitFound = true;
                }
            }
        }
        if (qubitFound) {
            return success();
        } else {
            return emitOpError("requires exactly one qubit");
        }
    }];
}

def QUIR_CallSubroutineOp : QUIR_Op<"call_subroutine", [DeclareOpInterfaceMethods<CallOpInterface>]> {
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
        OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", SymbolRefAttr::get(callee));
            $_state.addTypes(callee.getType().getResults());
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
        StringRef getCallee() { return callee(); }
        FunctionType getCalleeType();

        operand_iterator arg_operand_begin() { return operand_begin(); }
        operand_iterator arg_operand_end() { return operand_end(); }

    }];
}

def QUIR_CallKernelOp : QUIR_Op<"call_kernel", [DeclareOpInterfaceMethods<CallOpInterface>]> {
    let summary = "A call to a user-defined kernel function that needs to be specialized for input operands";
    let description = [{
        The `quir.call_kernel` operation represents calls to user-defined kernel function
        that needs to be specialized for its arguments. The callee function is attached to a symbol reference via
        an attribute. The arguments list must match the arguments provided by the callee. Kernels may only receive
        and return classical types.

        Example:
        ```mlir
        %majority = quir.call_kernel @majority_vote(%cbitarray) : (memref<3xi1>) -> i1
        ```
        This is valid only if the named kernel @majority_vote exists and takes a cbit array as arguments and
        returns an i1 as a result.
    }];

    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyClassical>:$operands);
    let results = (outs Optional<AnyClassical>:$res);

    let assemblyFormat = [{
        $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
    }];

    let builders = [
        OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", SymbolRefAttr::get(callee));
            $_state.addTypes(callee.getType().getResults());
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
}

def QUIR_CallCircuitOp : QUIR_Op<"call_circuit", [CallOpInterface, MemRefsNormalizable, DeclareOpInterfaceMethods<SymbolUserOpInterface>, DeclareOpInterfaceMethods<QubitOpInterface, ["getOperatedQubits"]>]> {
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
            $_state.addTypes(callee.getType().getResults());
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
        StringRef getCallee() { return callee(); }
        FunctionType getCalleeType();

        operand_range getArgOperands() {
            return {arg_operand_begin(), arg_operand_end()};
        }

        operand_iterator arg_operand_begin() { return operand_begin(); }
        operand_iterator arg_operand_end() { return operand_end(); }

        CallInterfaceCallable getCallableForCallee() {
            return (*this)->getAttrOfType<SymbolRefAttr>("callee");
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

def QUIR_CastOp : QUIR_Op<"cast", [NoSideEffect]> {
    let summary = "Cast between classical types";
    let description = [{
        The `quir.cast` operation represents a cast between different classical types, including non-QUIR types.

        Example:
        ```mlir
        %ang1 = "quir.cast"(%creg) : (memref<10xi1>) -> !quir.angle<20>
        ```
    }];

    let arguments = (ins AnyClassical:$arg);
    let results = (outs AnyClassical:$out);

    let hasCanonicalizer = 1;
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
        %dur1 = quir.declare_duration {value = "10ns"} : !quir.duration
        "quir.delay"(%dur1, %q1_1) : (!quir.duration, !quir.qubit)
        ```
    }];

    let arguments = (ins DurationOrStretch:$time, Variadic<AnyQubit>:$qubits);
    let results = (outs );

    let assemblyFormat = [{
        attr-dict $time `,` `(` $qubits `)` `:` type($time) `,` functional-type($qubits, results)
    }];
}

def QUIR_DelayCyclesOp : QUIR_Op<"delay_cycles", [
            UnitaryOp,
            DeclareOpInterfaceMethods<QubitOpInterface, ["getOperatedQubits"]>,
            NonInterferingNonDeadSideEffect]> {
    let summary = "Add a delay operation of a given number of cycles to a qubit";
    let description = [{
        The `quir.delay_cycles` operation represents a delay operation of the given
        number of cycles to a qubit, group of qubits, or all qubits (when no
        target qubit is given).

        Example:
        ```mlir
        "quir.delay_cycles"(%q1_1) { time = 1000 : i64}: (!quir.qubit) -> ()
        ```
    }];

    let arguments = (ins I64Attr:$time, Variadic<AnyQubit>:$qubits);
    let results = (outs );

    let assemblyFormat = [{
        `(` $qubits `)` attr-dict `:` functional-type($qubits, results)
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

// TODO: This op should be extracted to a system-level dialect
def QUIR_BroadcastOp : QUIR_Op<"broadcast", [NonInterferingNonDeadSideEffect]> {
    let summary = "Broadcast a value from this controller to all others";
    let description = [{
        The `quir.broadcast` operation represents a broadcast command that sends a value
        from this controller to all others. All other controllers should have a
        corresponding `quir.recv` operation. If only one controller should receive the
        message then the `quir.send` operation should be used instead.

        Example:
        ```mlir
        %angle1 = quir.constant #quir.angle<0.2 : !quir.angle<20>>
        quir.broadcast %angle1 : !quir.angle<20>
        ```
    }];

    let arguments = (ins AnyClassical:$val);

    let assemblyFormat = [{
        attr-dict $val `:` type($val)
    }];
}

// TODO: This op should be extracted to a system-level dialect
def QUIR_SendOp : QUIR_Op<"send", [NonInterferingNonDeadSideEffect]> {
    let summary = "Send a classical value from this controller to another";
    let description = [{
        The `quir.send` operation represents a send command from one controller to another.
        A corresponding `quir.recv` operation should receive the information sent.

        Example:
        ```mlir
        %cbit = "quir.measure"(%target) : (!quir.qubit<1>) -> i1
        quir.send %cbit : i1
        ```
    }];

    let arguments = (ins AnyClassical:$val, IndexAttr:$id);

    let assemblyFormat = [{
        attr-dict $val `to` $id `:` type($val)
    }];
}

// This should be in OpBase.td but for some reason isn't, maybe in LLVM 15?
def IndexArrayAttr : TypedArrayAttrBase<IndexAttr,
                                        "index array attribute"> {
  let constBuilderCall = "$_builder.getIndexArrayAttr($0)";
}

// TODO: This op should be extracted to a system-level dialect
def QUIR_RecvOp : QUIR_Op<"recv", [NonInterferingNonDeadSideEffect]> {
    let summary = "Receive classical values from other controllers";
    let description = [{
        The `quir.recv` operation represents a receive command for potentially
        multiple values. The fromIds array attribute indicates the Id of the
        sending control node for each independent value.

        Example:
        ```mlir
        %meas:2 = quir.recv {fromId = [0 : index, 1 : index]} : i1, i1
        ```
    }];

    let arguments = (ins OptionalAttr<IndexArrayAttr>:$fromIds);
    let results = (outs Variadic<AnyClassical>:$vals);

    let assemblyFormat = [{
        attr-dict `:` type($vals)
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
  RecursiveSideEffects
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
                       TypeAttr:$type,
                       OptionalAttr<StrAttr>:$sym_visibility);
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs
    )>];
  let extraClassDeclaration = [{

    /// Create a deep copy of this circuit and all of its blocks, remapping any
    /// operands that use values outside of the circuit using the map that is
    /// provided (leaving them alone if no entry is present). If the mapper
    /// contains entries for circuit arguments, these arguments are not
    /// included in the new circuit. Replaces references to cloned sub-values
    /// with the corresponding value that is copied, and adds those mappings to
    /// the mapper.
    CircuitOp clone(BlockAndValueMapping &mapper);
    CircuitOp clone();

    /// Clone the internal blocks and attributes from this circuit into dest.
    /// Any cloned blocks are appended to the back of dest. This circuit
    /// asserts that the attributes of the current circuit and dest are
    /// compatible.
    void cloneInto(CircuitOp dest, BlockAndValueMapping &mapper);

    FunctionType getType() {
        return getTypeAttr().getValue().cast<FunctionType>();
    }

    //===------------------------------------------------------------------===//
    // CallableOpInterface
    //===------------------------------------------------------------------===//

    /// Returns the region on the current operation that is callable. This may
    /// return null in the case of an external callable object, e.g. an external
    /// function.
    ::mlir::Region *getCallableRegion() { return isExternal() ? nullptr : &getBody(); }

    /// Returns the results types that the callable region produces when
    /// executed.
    ArrayRef<Type> getCallableResults() { return getType().getResults(); }

    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getType().getResults(); }

    /// Verify the type attribute of this function. Returns failure and emits
    /// an error if the attribute is invalid.
    LogicalResult verifyType() {
      auto type = getTypeAttr().getValue();
      if (!type.isa<FunctionType>())
        return emitOpError("requires '" + getTypeAttrName() +
                           "' attribute of function type");
      return success();
    }

    //===------------------------------------------------------------------===//
    // SymbolOpInterface Methods
    //===------------------------------------------------------------------===//

    bool isDeclaration() { return isExternal(); }
  }];
  let parser = [{ return ::parseCircuitOp(parser, result); }];
  let printer = [{ return ::print(*this, p); }];
  // TODO in LLVM 15 + this can become just let hasVerifier = 1;
  let verifier = [{ return ::verify(*this); }];
}

def QUIR_ReturnOp : QUIR_Op<"return", [HasParent<"CircuitOp">, NoSideEffect, Terminator, ReturnLike]>,
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
        let builders = [OpBuilder<(ins), [{ build($_builder, $_state, llvm::None); }]>];
        let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
        // TODO in LLVM 15 + this can become just let hasVerifier = 1;
        let verifier = [{ return ::verify(*this); }];
}

def QUIR_SwitchOp : QUIR_Op<"switch",
    [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
     RecursiveSideEffects,
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

    // TODO in LLVM 15 + this can become just let hasVerifier = 1;
    let verifier = [{ return ::verify(*this); }];
    let parser = [{ return ::parseSwitchOp(parser, result); }];
    let printer = [{ return ::print(p, *this); }];

}

def YieldOp : QUIR_Op<"yield", [NoSideEffect, ReturnLike, Terminator, ParentOneOf<["SwitchOp"]>]>{
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
    // TODO in LLVM 15 + this can become just let hasVerifier = 1;
    let verifier = [{ return ::verify(*this); }];
    let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
}


//===----------------------------------------------------------------------===//
// end Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//



#endif // QUIR_OPS
