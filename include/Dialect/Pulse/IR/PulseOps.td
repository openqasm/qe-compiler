//===- PulseOps.td - Pulse dialect ops ---------------------*- tablegen -*-===//
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

#ifndef PULSE_OPS
#define PULSE_OPS

include "Dialect/Pulse/IR/PulseTraits.td"

include "Dialect/QUIR/IR/QUIRTypeConstraints.td"
include "Dialect/Pulse/IR/PulseInterfaces.td"

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"


def Pulse_GaussianOp : Pulse_Op<"gaussian", [Pure, SequenceAllowed]> {
    let summary = "Create a Gaussian waveform.";
    let description = [{
        The `pulse.gaussian` operation creates a Gaussian waveform.
        Example:

        ```mlir
        %gauss = pulse.gaussian(%dur, %amp, %sigma) : (i32, complex<f64>, i32) -> !pulse.waveform
        ```
    }];

    let arguments = (ins AnyInteger:$dur, AnyComplex:$amp, AnyInteger:$sigma);
    let results = (outs Pulse_WaveformType:$wfr);

    let assemblyFormat = [{
        attr-dict `(` $dur `,` $amp `,` $sigma `)` `:` `(` type($dur) `,` type($amp) `,` type($sigma) `)` `->` type($wfr)
    }];

    let hasVerifier = 1;
}


def Pulse_GaussianSquareOp : Pulse_Op<"gaussian_square", [Pure, SequenceAllowed]> {
    let summary = "Create a Gaussian Square waveform.";
    let description = [{
        The `pulse.gaussian_square` operation creates a Gaussian Square waveform. It is a Gaussian
        on either end with a constant pulse of amplitude `amp` and duration `width` in the center.
        Example:

        TODO: The width parameter should be renamed to something more descriptive.

        ```mlir
        %gauss_sq = pulse.gaussian_square(%dur, %amp, %sigma, %width) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
        ```
    }];

    let arguments = (ins AnyInteger:$dur, AnyComplex:$amp, AnyInteger:$sigma, AnyInteger:$width);
    let results = (outs Pulse_WaveformType:$wfr);

    let assemblyFormat = [{
        attr-dict `(` $dur `,` $amp `,` $sigma `,` $width `)` `:` `(` type($dur) `,` type($amp) `,` type($sigma) `,` type($width) `)` `->` type($wfr)
    }];

    let hasVerifier = 1;

}

def Pulse_DragOp : Pulse_Op<"drag", [Pure, SequenceAllowed]> {
    let summary = "Create a Drag waveform.";
    let description = [{
        The `pulse.drag` operation creates a DRAG waveform. The DRAG waveform is composed of a
        Gaussian and a Gaussian derivative. It is designed to suppress the 1->2 frequency transition,
        thereby reducing leakage into the |2> state. The formal expression is
        gaussian + i*beta*d/dx [gaussian], where i is the imaginary number.

        Example:

        ```mlir
        %drag = pulse.drag(%dur, %amp, %sigma, %beta) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
        ```
    }];

    let arguments = (ins AnyInteger:$dur, AnyComplex:$amp, AnyInteger:$sigma, F64:$beta);
    let results = (outs Pulse_WaveformType:$wfr);

    let assemblyFormat = [{
        attr-dict `(` $dur `,` $amp `,` $sigma `,` $beta `)` `:` `(` type($dur) `,` type($amp) `,` type($sigma) `,` type($beta) `)` `->` type($wfr)
    }];

    let hasVerifier = 1;
}

def Pulse_ConstOp : Pulse_Op<"const_waveform", [Pure, SequenceAllowed]> {
    let summary = "Create a Constant waveform.";
    let description = [{
        The `pulse.const_waveform` operation creates a constant waveform. This is a flat top waveform
        with all values equal to some constant, complex amplitude.

        Example:

        ```mlir
        %const = pulse.const_waveform(%dur, %amp) : (i32, complex<f64>) -> !pulse.waveform
        ```
    }];

    let arguments = (ins AnyInteger:$dur, AnyComplex:$amp);
    let results = (outs Pulse_WaveformType:$wfr);

    let assemblyFormat = [{
        attr-dict `(` $dur `,` $amp `)` `:` `(` type($dur) `,` type($amp) `)` `->` type($wfr)
    }];

    let hasVerifier = 1;
}

def Pulse_Waveform_CreateOp : Pulse_Op<"create_waveform", [Pure, SequenceAllowed,
    DeclareOpInterfaceMethods<PulseOpSchedulingInterface, ["getDuration"]>]> {
    let summary = "Creates a pulse waveform from a list of complex numbers.";
    let description = [{
        The `pulse.waveform` operation creates a waveform from a list of complex numbers. The complex
        numbers are modeled as a 2 element list `[real, imag]`."""

        Example:

        ```mlir
        %wfr = pulse.create_waveform dense<[[0.0, 0.5], [0.5, 0.5], [0.5, 0.0]]> : tensor<3x2xf64> -> !pulse.waveform
        ```
    }];

    let arguments = (ins F64ElementsAttr:$samples);
    let results = (outs Pulse_WaveformType:$wfr);

    let assemblyFormat = [{
        attr-dict $samples `->` type($wfr)
    }];

    let hasVerifier = 1;
}

def Pulse_WaveformContainerOp : Pulse_Op<"waveform_container", [Pure, NoTerminator]> {
    let summary = "Creates a container for pulse waveforms.";
    let description = [{
        The `pulse.waveform_container` is a container for `pulse.create_waveform` operations. All the `pulse.create_waveform`
        operations must have a `pulse.waveformName` attribute.

        Example:

        ```mlir
        pulse.waveform_container {
        %0 = pulse.create_waveform {pulse.waveformName = "X90"} dense<[[0.0, 0.5], [0.5, 0.5], [0.5, 0.0]]> : tensor<3x2xf64> -> !pulse.waveform
        %1 = pulse.create_waveform {pulse.waveformName = "Y90"} dense<[[0.1, 0.5], [0.7, 0.5], [0.1, 0.0]]> : tensor<3x2xf64> -> !pulse.waveform
        }
        ```
    }];

    let regions = (region AnyRegion:$body);

    let assemblyFormat = [{
        attr-dict-with-keyword $body
    }];

    let hasVerifier = 1;
}

def Pulse_Frame_CreateOp : Pulse_Op<"create_frame", [Pure, SequenceAllowed, SameVariadicOperandSize]> {
    let summary = "Creates a frame with a frequency and a phase.";
    let description = [{
        The `pulse.frame` operation creates a frame with optional arguments to initialize amplitude, frequency, and phase. One can also
        set/shift the amplitude, frequency, and phase of a frame using pulse Dialect operations.

        Example:

        ```mlir
        %frame = pulse.frame() {uid = "a unique identifier"} : () -> !pulse.frame
        ```
    }];

    let arguments = (ins StrAttr:$uid, Optional<AnyComplex>:$initial_amp, Optional<F64>:$initial_freq, Optional<AnyAngle>:$initial_phase);
    let results = (outs Pulse_FrameType:$out);
}

def Pulse_Port_CreateOp : Pulse_Op<"create_port", [Pure]> {
    let summary = "Creates a port for stimulus pulses or acquiring measurement output.";
    let description = [{
        The `pulse.create_port` operation creates a port which acts as a host for
        the underlying frames that are then used for playing and capturing operations.

        Ports must be created and declared outside of a pulse.sequence as they are the
        interface that couples a classical program to the invocation of a real-time
        pulse sequence.

        Example:

        ```mlir
        %port = pulse.getPort() {uid = "a unique identifier"} : () -> !pulse.port
        ```
    }];

    let arguments = (ins StrAttr:$uid);
    let results = (outs Pulse_PortType:$out);
}

def Pulse_MixFrameOp : Pulse_Op<"mix_frame", [Pure, SequenceAllowed, SameVariadicOperandSize]> {
    let summary = "Mixes a frame with a port.";
    let description = [{
        The `pulse.mix_frame` operation creates a frame mixed with a port, with optional arguments to initialize amplitude, frequency, and phase. One can also
        set/shift the amplitude, frequency, and phase of a mixed frame using pulse Dialect operations.

        The difference between `pulse.frame` and `pulse.mix_frame` is that `pulse.frame` is defined at the logical level, but the `pulse.mix_frame` is defined
        at the physical level and is specific to a target hardware. One can define the pulse program at the logical level using `pulse.frame` operations and
        then replace the `pulse.frame` operations with `pulse.mix_frame` operations in a compiler pass according to the target hardware configuration.

        Example:

        ```mlir
        pulse.mix_frame(%port) {uid = "a unique identifier"} : (!pulse.port) -> (!port.mixed_frame)
        ```
    }];

    let arguments = (ins Pulse_PortType:$port, StrAttr:$uid, Optional<AnyComplex>:$initial_amp, Optional<F64>:$initial_freq, Optional<AnyAngle>:$initial_phase);
    let results = (outs Pulse_MixedFrameType:$out);
}

def Pulse_Kernel_CreateOp : Pulse_Op<"create_kernel", [Pure, SequenceAllowed]> {
    let summary = "Creates a kernel which defines the integration window.";
    let description = [{
        The `pulse.create_kernel` operation creates  a kernel which defines
        the integration window and potential further processing such as rotation and discrimination.
        Example:

        ```mlir
        %kernel = pulse.create_kernel (%waveform) : !pulse.kernel
        ```
    }];

    let arguments = (ins Pulse_WaveformType:$weights);
    let results = (outs Pulse_KernelType:$out);

    let assemblyFormat = [{
        attr-dict `(` $weights `)` `:` `(` type($weights) `)` `->` type($out)
    }];
}

def Pulse_PlayOp : Pulse_Op<"play", [SequenceRequired, HasTargetFrame,
    DeclareOpInterfaceMethods<PulseOpSchedulingInterface, ["getDuration"]>]> {
    let summary = "Play a waveform on a given frame.";
    let description = [{
        The `pulse.play` operation plays a waveform on a frame,
        optional attributes include: angle (in radians), ssb (frequency in Hz),
        amplitude (float 64).

        Examples:

        ```mlir
        pulse.play(%frame, %wfr) : (!pulse.frame, !pulse.waveform)
        pulse.play(%mixed_frame, %wfr) : (!pulse.mixed_frame, !pulse.waveform)
        ```
    }];

    let arguments = (ins AnyFrame:$target, Pulse_WaveformType:$wfr,
                     OptionalAttr<F64Attr>:$angle,
                     OptionalAttr<F64Attr>:$ssb,
                     OptionalAttr<F64Attr>:$amplitude);

    let builders = [
        OpBuilder<(ins "Value":$target, "Value":$wfr), [{
            // dummy builder for QUIRToPulsePass
            // this allows existing QUIRToPulsePass to compile
            // TODO: Implement as part of QUIRToPulse re-implementation
            //       with new Pulse updates
        }]>
        ];

    let assemblyFormat = [{
        attr-dict `(` $target  `,` $wfr `)` `:` `(` type($target) `,` type($wfr) `)`
    }];

    let extraClassDeclaration = [{
        // get a string hash of the waveform and target being played
        llvm::Expected<std::string> getWaveformHash(CallSequenceOp callOp);
    }];

    let hasVerifier = 1;
}

def Pulse_SetFrequencyOp : Pulse_Op<"set_frequency", [SequenceRequired, HasTargetFrame]> {
    let summary = "Set a frequency of a frame.";
    let description = [{
        The `pulse.set_frequency` operation sets the frequency of a frame.

        TODO: Better specify the timing semantics of when the frequency modifications occurrs as this has
        a strong impact on phase accumulation in frames.

        Example:

        ```mlir
        pulse.set_frequency(%frame, %frequency) : (!pulse.frame, f64)
        pulse.set_frequency(%mixed_frame, %frequency) : (!pulse.mixed_frame, f64)
        ```
    }];

    let arguments = (ins AnyFrame:$target, F64:$frequency);

    let assemblyFormat = [{
        attr-dict `(` $target  `,` $frequency `)` `:` `(` type($target) `,` type($frequency) `)`
    }];

    let extraClassDeclaration = [{
        double getFreqFromDefiningOp() {
            // get op that defines frequency and return its value
            auto freqConstOp = dyn_cast<mlir::arith::ConstantFloatOp>((*this).getFrequency().getDefiningOp());
            assert (freqConstOp && "assume frequency to be defined by ConstantFloatOp");
            return freqConstOp.value().convertToDouble();
        }
    }];

    let hasVerifier = 1;
}

def Pulse_ShiftFrequencyOp : Pulse_Op<"shift_frequency", [SequenceRequired, HasTargetFrame]> {
    let summary = "Shift the frequency of a frame.";
    let description = [{
        The `pulse.shift_frequency` operation shifts the frequency of a frame.

        TODO: Better specify the timing semantics of when the frequency modifications occurs as this has
        a strong impact on phase accumulation in frames.

        Example:

        ```mlir
        pulse.shift_frequency(%frame, %frequencyOffset) : (!pulse.frame, f64)
        pulse.shift_frequency(%mixed_frame, %frequencyOffset) : (!pulse.mixed_frame, f64)
        ```
    }];

    let arguments = (ins AnyFrame:$target, F64:$frequencyOffset);

    let assemblyFormat = [{
        attr-dict `(` $target  `,` $frequencyOffset `)` `:` `(` type($target) `,` type($frequencyOffset) `)`
    }];

    let extraClassDeclaration = [{
        double getFreqOffsetFromDefiningOp() {
            // get op that defines frequency offset and return its value
            auto freqOffsetConstOp = dyn_cast<mlir::arith::ConstantFloatOp>((*this).getFrequencyOffset().getDefiningOp());
            assert (freqOffsetConstOp && "assume frequency offset to be defined by ConstantFloatOp");
            return freqOffsetConstOp.value().convertToDouble();
        }
    }];

    let hasVerifier = 1;
}

def Pulse_SetPhaseOp : Pulse_Op<"set_phase", [SequenceRequired, HasTargetFrame]> {
    let summary = "Set the phase of a frame.";
    let description = [{
        The `pulse.set_phase` operation sets the phase of a frame.

        TODO: Reconsider whether we should use quir angles or floats.
        TODO: Better specify the timing semantics of when the phase modifications occurrs.

        Example:

        ```mlir
        pulse.set_phase(%frame, %phase) : (!pulse.frame, f64)
        pulse.set_phase(%mixed_frame, %phase) : (!pulse.mixed_frame, f64)
        ```
    }];

    let arguments = (ins AnyFrame:$target, F64:$phase);

    let assemblyFormat = [{
        attr-dict `(` $target  `,` $phase `)` `:` `(` type($target) `,` type($phase) `)`
    }];

    let extraClassDeclaration = [{
        double getPhaseFromDefiningOp() {
            // get op that defines phase and return its value
            auto phaseConstOp = dyn_cast<mlir::arith::ConstantFloatOp>((*this).getPhase().getDefiningOp());
            assert (phaseConstOp && "assume phase to be defined by ConstantFloatOp");
            return phaseConstOp.value().convertToDouble();
        }
    }];

    let hasVerifier = 1;
}

def Pulse_ShiftPhaseOp : Pulse_Op<"shift_phase", [SequenceRequired, HasTargetFrame]> {
    let summary = "Shift a phase of a frame.";
    let description = [{
        The `pulse.shift_phase` operation shifts the phase of a frame.

        TODO: Better specify the timing semantics of when the phase modifications occurrs.
        TODO: Reconsider whether we should use quir angles or floats.

        Example:

        ```mlir
        pulse.shift_phase(%frame, %phaseOffset) : (!pulse.frame, f64)
        pulse.shift_phase(%mixed_frame, %phaseOffset) : (!pulse.mixed_frame, f64)
        ```
    }];

    let arguments = (ins AnyFrame:$target, F64:$phaseOffset);

    let assemblyFormat = [{
        attr-dict `(` $target  `,` $phaseOffset `)` `:` `(` type($target) `,` type($phaseOffset) `)`
    }];

    let extraClassDeclaration = [{
        double getPhaseFromDefiningOpOffset() {
            // get op that defines phase offset and return its value
            auto phaseOffsetConstOp = dyn_cast<mlir::arith::ConstantFloatOp>((*this).getPhaseOffset().getDefiningOp());
            assert (phaseOffsetConstOp && "assume phase offset to be defined by ConstantFloatOp");
            return phaseOffsetConstOp.value().convertToDouble();
        }
    }];

    let hasVerifier = 1;
}

def Pulse_SetAmplitudeOp : Pulse_Op<"set_amplitude", [SequenceRequired, HasTargetFrame]> {
let summary = "Set a scaling factor for pulses of a frame.";
    let description = [{
        The `pulse.set_amplitude` operation sets a scaling factor for pulses of a frame.

        Example:

        ```mlir
        pulse.set_amplitude(%frame, %amp) : (!pulse.frame, f64)
        pulse.set_amplitude(%mixed_frame, %amp) : (!pulse.mixed_frame, f64)
        ```
    }];

    let arguments = (ins AnyFrame:$target, F64:$amp);

    let assemblyFormat = [{
        attr-dict `(` $target  `,` $amp `)` `:` `(` type($target) `,` type($amp) `)`
    }];

    let extraClassDeclaration = [{
        double getAmpFromDefiningOp() {
            // get op that defines amp and return its value
            auto ampConstOp = dyn_cast<mlir::arith::ConstantFloatOp>((*this).getAmp().getDefiningOp());
            assert (ampConstOp && "assume amplitude to be defined by ConstantFloatOp");
            return ampConstOp.value().convertToDouble();
        }
    }];

    let hasVerifier = 1;
}

def Pulse_CaptureOp : Pulse_Op<"capture", [SequenceRequired, HasTargetFrame]> {
    let summary = "Capture the incoming port and use a kernel for integration.";
    let description = [{
        The `pulse.capture` operation captures the incoming data on a frame using a kernel.

        TODO: This needs to be reworked with timing and other relevant information for hardware captures.
        TODO: Kernels are excessively coupled to current readout approaches and likely should not be an
        operand but rather an attribute for specific hardware backends.

        Example:

        ```mlir
        pulse.capture(%frame) : (!pulse.frame) -> i1
        pulse.capture(%mixed_frame) : (!pulse.mixed_frame) -> i1
        ```
    }];

    let arguments = (ins AnyFrame:$target);
    let results = (outs I1:$out);

    let assemblyFormat = [{
        attr-dict `(` $target `)` `:` `(` type($target) `)` `->` type($out)
    }];

    let hasVerifier = 1;
}

def Pulse_DelayOp : Pulse_Op<"delay", [SequenceAllowed, HasTargetFrame]> {
    let summary = "Delay by an integer duration of samples on a set of pulse frames.";
    let description = [{
        The `pulse.delay` operation represents a delay operation of integer length on a set of pulse
        frames.

        TODO: The frame operand should be before the delay duration.
        TODO: Durations should be greater precision than i32.
        TODO: Durations should be typed and integrated into a timing system.

        Example:
        ```mlir
        pulse.delay(%frame, %dur) : (!pulse.frame, i32)
        pulse.delay(%mixed_frame, %dur) : (!pulse.mixed_frame, i32)
        ```
    }];

    let arguments = (ins AnyFrame:$target,  AnyInteger:$dur);

    let assemblyFormat = [{
        attr-dict `(` $target `,` $dur `)` `:` `(` type($target) `,` type($dur) `)`
    }];

    let hasVerifier = 1;
}

def Pulse_BarrierOp : Pulse_Op<"barrier", [Pure, SequenceAllowed]> {
    let summary = "Add a barrier operation on a set of pulse frames.";
    let description =   [{
        The `pulse.barrier` operation synchronizes the timing of all frames. It does this by inserting
        delays until the maximum time amongst all frames is reached. It therefore obeys a "push forward"
        rule.

        TODO: Barriers should integrate into a timing layer.
        TODO: Barriers should have the same assembly format as other pulse instructions.

        Example:
        ```mlir
        pulse.barrier %d_frame : !pulse.frame // barrier frame
        pulse.barrier %d_frame, %cr_frame : !pulse.frame, !pulse.frame // barrier multiple frames
        ```
    }];

    let arguments = (ins Variadic<AnyFrame>:$frames);

    let assemblyFormat = [{
        attr-dict $frames `:` type($frames)
    }];
}

def Pulse_MixOp : Pulse_Op<"mix", [SequenceAllowed]> {
    let summary = "Mix (multiply) two waveforms together. The waveforms should be of the same duration.";
    let description = [{
        The `pulse.mix` operation multiplies two waveforms together, ie `f(t_i) = f1(t_i)*f2(t_i)` for
        two input waveforms `f1(t)`, `f2(t)`.

        Example:

        ```mlir
        %wfr = pulse.mix(%wfr1, %wfr2) : (!pulse.waveform, !pulse.waveform) -> !pulse.waveform
        ```
    }];

    let arguments = (ins Pulse_WaveformType:$wfr1, Pulse_WaveformType:$wfr2);
    let results = (outs Pulse_WaveformType:$wfr);

    let assemblyFormat = [{
        attr-dict `(` $wfr1  `,` $wfr2 `)` `:` `(` type($wfr1) `,` type($wfr2) `)` `->` type($wfr)
    }];
}


def Pulse_SumOp : Pulse_Op<"sum", [SameOperandsAndResultType, SequenceAllowed]> {
    let summary = "Add pulse waveforms sample by sample. The waveforms must be the same length.";
    let description = [{
        The `pulse.sum` operation adds two waveforms together, ie `s(t_i) = s1(t_i)+s2(t_i)` for
        two input waveforms `s1(t)`, `s2(t)`.

        Example:

        ```mlir
        %new_wfr = pulse.sum(%wfr0, %wfr1) : (!pulse.waveform, !pulse.waveform): -> !pulse.waveform
        ```
    }];

    let arguments = (ins Pulse_WaveformType:$wfr0, Pulse_WaveformType:$wfr1);
    let results = (outs Pulse_WaveformType:$new_wfr);

    let assemblyFormat = [{
        attr-dict `(` $wfr0 `,` $wfr1 `)`  `:` `(` type($wfr0) `,` type($wfr1) `)` `->` type($new_wfr)
    }];
}

def Pulse_CallSequenceOp : Pulse_Op<"call_sequence", [CallOpInterface, MemRefsNormalizable, DeclareOpInterfaceMethods<SymbolUserOpInterface>, SequenceAllowed]> {
    let summary = "Call a pulse sequence operation";
    let description = [{
        The `pulse.call_sequence` operation represents calls to launch a pulse sequence in the target system.
        The callee function is attached to a symbol reference via an attribute. The arguments list must match
        the argument types provided by the callee. The calling of a sequence represents the classical<->real-time
        pulse sequencing interaction within the specified program. All classical input values to the sequence
        should be transferred before the invocation of the sequence routine to enable deterministic and
        real-time execution of the pulse sequence in hardware.

        Example:
        ```mlir
        %classical_result = pulse.call_sequence @waveform_sequence(%theta) : (quir.angle<32>) -> (i1, i1)
        ```
    }];

    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyTypeOf<[AnyClassical, AnyPulse]>>:$operands);
    let results = (outs Variadic<AnyClassical>:$res);

    let assemblyFormat = [{
        $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
    }];

    let builders = [
        OpBuilder<(ins "SequenceOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee",SymbolRefAttr::get(callee));
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

//===----------------------------------------------------------------------===//
//
// This code section was derived and modified from the LLVM project
// Consequently it is licensed as described below.
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

def Pulse_SequenceOp : Pulse_Op<"sequence", [
  AutomaticAllocationScope, CallableOpInterface,
  DeclareOpInterfaceMethods<PulseOpSchedulingInterface, ["getDuration"]>,
  FunctionOpInterface, IsolatedFromAbove, Symbol, SequenceAllowed
]> {
  let summary = "An operation with a name containing a single `SSACFG` region corresponding to a pulse sequence execution";
  let description = [{

    Defines a sequence(function) operation representing a
    deterministically timed pulse sequence execution
    on the QPU *without* any classical compute or control-flow.
    Effectively corresponding to a callable "pulse sequence block".

    The SequenceOp is modelled after the FuncOp of the MLIR func
    dialect.

    A sequence may only contain operations legally allowed by the pulse
    dialect. All pulse-type operations should in-turn exist within regions
    contained by `pulse.sequence` operations.

    All non-constant classical values required to define the sequence must
    be provided as `pulse.sequence` arguments to allow a clean separation of
    classical, quantum and pulse value-passing processes within the MLIR.

    Pulse programs should be canonicalized such that all pulse
    operations occurr within a region contained by a `pulse.sequence` operation
    and are reached through a corresponding `pulse.call_sequence`.

    Operations within the sequence cannot implicitly capture values defined
    outside of the sequence, i.e. they are `IsolatedFromAbove`. All
    external references must use sequence arguments or attributes that establish
    a symbolic connection (e.g. symbols referenced by name via a string
    attribute like SymbolRefAttr). An external sequence declaration (used when
    referring to a sequence declared in some other module) has no body. While
    the MLIR textual form provides a nice inline syntax for sequence arguments,
    they are internally represented as “block arguments” to the first block in
    the region.

    Syntax:

    ```
    op ::= `pulse.sequence` symbol-ref-id `(` argument-list `)` (`->`
    function-result-list)? function-attributes? region
    ```

    Example:

    ```mlir
    // External sequence function definitions.
    pulse.sequence @x()

    // A function that returns its argument twice:
    pulse.sequence @waveform_sequence(%angle: quir.angle<32>) -> i1 {
      pulse.return %result: i1
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
    static SequenceOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs = {});
    static SequenceOp create(Location location, StringRef name, FunctionType type,
                         Operation::dialect_attr_range attrs);
    static SequenceOp create(Location location, StringRef name, FunctionType type,
                         ArrayRef<NamedAttribute> attrs,
                         ArrayRef<DictionaryAttr> argAttrs);

    /// Create a deep copy of this sequence and all of its blocks, remapping any
    /// operands that use values outside of the sequence using the map that is
    /// provided (leaving them alone if no entry is present). If the mapper
    /// contains entries for sequence arguments, these arguments are not
    /// included in the new sequence. Replaces references to cloned sub-values
    /// with the corresponding value that is copied, and adds those mappings to
    /// the mapper.
    SequenceOp clone(IRMapping &mapper);
    SequenceOp clone();

    /// Clone the internal blocks and attributes from this sequence into dest.
    /// Any cloned blocks are appended to the back of dest. This sequence
    /// asserts that the attributes of the current sequence and dest are
    /// compatible.
    void cloneInto(SequenceOp dest, IRMapping &mapper);

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

def Pulse_ReturnOp : Pulse_Op<"return", [Pure, Terminator, ReturnLike, SequenceRequired]>,
    Arguments<(ins Variadic<AnyClassical>:$operands)>, Results<(outs)> {
        let summary = "Terminator for pulse.sequence.";
        let description = [{
                A terminator for regions that appear in the body of the `pulse.sequence`
                sequence operation. The operands to the `pulse.return` are the result values
                that are returned to the `pulse.call_sequence` invocation of the target sequence.

                Example:

                ```mlir
                pulse.sequence @foo(%angle: quir.angle<32>) -> (i1, i1)
                    ...
                    pulse.return %0, %1: i1, i1
                }
            }];
        let builders = [OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>];
        let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
        let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// end Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//

#endif // PULSE_OPS
