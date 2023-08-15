### Example python script for generating mlir for a variety of Pulse Ops

from qss_compiler.mlir.ir import *
from qss_compiler.mlir.dialects import arith, builtin, std
from qss_compiler.mlir.dialects import pulse, complex, quir

import numpy as np

with Context(), Location.unknown():
    # register dialects
    pulse.pulse.register_dialect()
    quir.quir.register_dialect()

    # declare types
    i1 = IntegerType.get_signless(1)
    i32 = IntegerType.get_signless(32)
    f64 = F64Type.get()

    c64 = ComplexType.get(f64)

    p = pulse.PortType.get()
    wf = pulse.WaveformType.get()
    mf = pulse.MixedFrameType.get()
    f = pulse.FrameType.get()
    k = pulse.KernelType.get()

    # create module
    module = Module.create()
    with InsertionPoint(module.body):
        # create main func op
        func = builtin.FuncOp("main", ([],[i1, i1]))
        func.add_entry_block()

        # create test sequence op
        seq = pulse.SequenceOp("test_pulse_ops", [p,p,f,mf,c64], [i1,i1])
        seq.add_entry_block()
    # define sequence
    with InsertionPoint(seq.entry_block):
        # define args
        p0, p1, f0, mf0, amp = seq.entry_block.arguments

        # # define constants
        dur = arith.ConstantOp(i32, 160)
        sigma = arith.ConstantOp(i32, 40)
        width = arith.ConstantOp(i32, 1000)
        beta = arith.ConstantOp(f64, -1.3677586253287046)

        samples = np.array([[0.0,0.5],[0.5,0.5],[0.5,0.0]])
        samples_2d = DenseElementsAttr.get(samples)

        # # create waveforms
        gauss = pulse.GaussianOp(wf, dur, amp, sigma)
        gauss_square = pulse.GaussianSquareOp(wf, dur, amp, sigma, width)
        drag = pulse.DragOp(wf, dur, amp, sigma, beta)
        const = pulse.ConstOp(wf, dur, amp)
        kernel_waveform = pulse.Waveform_CreateOp(wf, samples_2d)

        # mixed frame
        mf1 = pulse.MixFrameOp(p0,"mf1-p0")

        # define complex amp
        param_amp_r = arith.ConstantOp(f64, 0.10086211860780928)
        param_amp_i = arith.ConstantOp(f64, 0.0012978777572167797)
        param_amp = complex.CreateOp(param_amp_r, param_amp_i)

        # freq stuff
        # define freq
        fc = arith.ConstantOp(f64, 200.e4)

        pulse.SetFrequencyOp(mf0, fc)
        pulse.SetFrequencyOp(f0, fc)

        pulse.ShiftFrequencyOp(mf0, fc)
        pulse.ShiftFrequencyOp(f0, fc)

        # phase stuff
        # define angle
        angle = arith.ConstantOp(f64, 3.14)

        pulse.SetPhaseOp(mf0, fc)
        pulse.SetPhaseOp(f0, fc)

        pulse.ShiftPhaseOp(mf0, fc)
        pulse.ShiftPhaseOp(f0, fc)

        # barrier
        pulse.BarrierOp([f0])

        # delay
        delay_dur = arith.ConstantOp(i32, 100)
        pulse.DelayOp(mf0, delay_dur)

        # play
        pulse.PlayOp(mf0, drag)

        # # kernel
        kernel = pulse.Kernel_CreateOp(k, kernel_waveform)
        res0 = pulse.CaptureOp(i1, mf0)
        pulse.ReturnOp([res0, res0])

    # define main func
    with InsertionPoint(func.entry_block):
        # create ports
        p0 = pulse.Port_CreateOp("p0")
        p1 = pulse.Port_CreateOp("p1")

        # create frame and mixed frame
        f0 = pulse.Frame_CreateOp("f0")
        mf0 = pulse.MixFrameOp(p0, "mf0-p0")

        amp_r = arith.ConstantOp(f64, 0.10086211860780928)
        amp_i = arith.ConstantOp(f64, 0.0012978777572167797)
        amp = complex.CreateOp(amp_r, amp_i)

        # call test sequence
        res = pulse.CallSequenceOp([i1,i1], "test_pulse_ops", [p0, p1, f0, mf0, amp])
        std.ReturnOp(res)

print(module)

# writing to a file
print("Writing to test.mlir")
test = open("TestOps.mlir", "w")
test.write(str(module))
test.close()
print("Done!")