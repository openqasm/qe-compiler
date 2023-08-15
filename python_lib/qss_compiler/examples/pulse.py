#!/usr/bin/env python
# coding: utf-8

# Using this notebook requires manual install of MLIR python bindings:
# 1. Copy or link to `qss-compiler/third_party/mlir/pyproject.toml` from `~/.conan/data/llvm/qss/stable/<hash>/python_packages/mlir_core`
# 2. Run `pip install -e .`
# 3. Copy or link to `~/.conan/data/llvm/qss/stable/<hash>/python_packages/mlir_core/mlir/_mlir_libs/libMLIRPythonCAPI.dylib` from `<python>/lib`
# 4. build compiler
# 5. cd `qss_compiler/python_lib`
# 6. run `bash setup_mlir.sh`
# 7. run `pip install -e .`

from qss_compiler.mlir.ir import Context, InsertionPoint, Location, Module
from qss_compiler.mlir.ir import F64Type, IntegerType, IndexType
from qss_compiler.mlir.dialects import arith, builtin, std, scf

from qss_compiler.mlir.dialects import pulse, quir, complex # noqa: F401, E402

import numpy as np


num_samples = 3
samples = np.arange(0,num_samples) + np.arange(0+1,num_samples+1) * 1j
samples_2d = np.zeros((samples.shape[0],2))
samples_2d[:,0] = np.real(samples)
samples_2d[:,1] = np.imag(samples)


with Context() as ctx:
    
    pulse.pulse.register_dialect()
    quir.quir.register_dialect()

    f64 = F64Type.get(ctx)
    i32 = IntegerType.get_signless(32)
    i1 = IntegerType.get_signless(1)

    idx = IndexType.get()
    
    mf = pulse.MixedFrameType.get(ctx)
    wf = pulse.WaveformType.get(ctx)
    
    with Location.unknown(ctx) as loc:
        module = Module.create(loc)

        with InsertionPoint(module.body):
            func = builtin.FuncOp("main", ([], [i32]))
            func.add_entry_block()
            
        with InsertionPoint(module.body):
            
            seq1 = pulse.SequenceOp("seq_1",[wf, wf, mf, mf],[i1])
            seq1.add_entry_block()
            
    #     with InsertionPoint(seq1.entry_block):
    #         gs, dg, mf1, mf2 = seq1.arguments
    #         c0 = arith.ConstantOp(i32, 656)
    #         pulse.DelayOp(mf1, c0)
    #         pulse.PlayOp(mf1, dg, 1.4, 5.25, 1.0)
    #         zero = arith.ConstantOp(i1, 0)
    #         ret = pulse.ReturnOp(zero)

            
        with InsertionPoint(func.entry_block):
    #         quir.SystemInitOp()

            c0 = arith.ConstantOp(f64, 0.0)
            c1_r = arith.ConstantOp(f64, 0.0)
            c1_i = arith.ConstantOp(f64, 0.0)
            c1_c = complex.CreateOp(c1_r, c1_i)

    #         ph0 = quir.ConstantOp("angle", 0.0)
            p0 = pulse.Port_CreateOp("Q0")
            
    #         fM0 = pulse.Frame_CreateOp(c1_c, c0, ph0)
    #         mfM0 = pulse.MixFrameOp("drive" ,p0, fM0)

    #         fMTRIGOUT = pulse.Frame_CreateOp(c1_c, c0, ph0)
    #         mfMTRIGOUT = pulse.MixFrameOp("measure", p0, fMTRIGOUT)
            
    #         c2 = arith.ConstantOp(idx, 0)
    #         c3 = arith.ConstantOp(idx, 1000)
    #         c4 = arith.ConstantOp(idx, 1)
            
    #         loop = quir.ShotLoop(c2,c3,c4)
            
    #         with InsertionPoint(loop.body):
    #             dur = quir.ConstantOp("duration","150us")
    #             quir.DelayOp(dur)
    #             si = quir.ShotInitOp(1000, False)
    #             gaussian_square_ice0 = pulse.Waveform_CreateOp(samples_2d)
    #             drag0 = pulse.Waveform_CreateOp(2* samples_2d)
                
    #             res0 = pulse.CallSequenceOp([i1], "seq_1", [drag0, gaussian_square_ice0, mfM0, mfMTRIGOUT ])
    #             scf.YieldOp(loop.inner_iter_args)
            
    #         quir.SystemFinalizeOp()
            zero = arith.ConstantOp(i32, 0)
            std.ReturnOp(zero)
            
print(str(module))


