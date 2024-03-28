#!/usr/bin/env python
# coding: utf-8

# (C) Copyright IBM 2024.
#
# This code is part of Qiskit.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from qss_compiler.mlir.ir import Context, InsertionPoint, Location, Module
from qss_compiler.mlir.ir import F64Type, IntegerType, IndexType, ComplexType
from qss_compiler.mlir.dialects import arith, func, complex

from qss_compiler.mlir.dialects import pulse, quir, qcs

import numpy as np


num_samples = 3
samples = np.arange(0, num_samples) + np.arange(0 + 1, num_samples + 1) * 1j
samples_2d = np.zeros((samples.shape[0], 2))
samples_2d[:, 0] = np.real(samples)
samples_2d[:, 1] = np.imag(samples)


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
            mainFunc = func.FuncOp("main", ([], [i32]))
            mainFunc.add_entry_block()

        with InsertionPoint(module.body):
            seq1 = pulse.SequenceOp("seq_1", [wf, wf, mf, mf], [i1])
            seq1.add_entry_block()

        with InsertionPoint(seq1.entry_block):
            gs, dg, mf1, mf2 = seq1.arguments
            c0 = arith.ConstantOp(i32, 656)
            pulse.DelayOp(mf1, c0)

            amp = arith.ConstantOp(f64, 1.4)
            sigma = arith.ConstantOp(f64, 5.25)
            beta = arith.ConstantOp(f64, 1.0)
            drag = pulse.DragOp(dg, amp, sigma, beta)
            pulse.PlayOp(mf1, drag)

            zero = arith.ConstantOp(i1, 0)
            ret = pulse.ReturnOp(zero)

        with InsertionPoint(mainFunc.entry_block):
            qcs.SystemInitOp()

            c0 = arith.ConstantOp(f64, 0.0)
            c1_r = arith.ConstantOp(f64, 0.0)
            c1_i = arith.ConstantOp(f64, 0.0)
            c1_c = complex.CreateOp(ComplexType.get(f64), c1_r, c1_i)

            p0 = pulse.Port_CreateOp("Q0")

            qcs.SystemFinalizeOp()
            zero = arith.ConstantOp(i32, 0)
            func.ReturnOp(zero)

print(str(module))
