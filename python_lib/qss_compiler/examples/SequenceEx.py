#!/usr/bin/env python
#
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

# A simple example of creating an SequenceOp using the pulse python bindings.

from qss_compiler.mlir.ir import InsertionPoint, Location, Module, Context
from qss_compiler.mlir.ir import IntegerType, F64Type, ComplexType

from qss_compiler.mlir.dialects import arith, complex

from qss_compiler.mlir.dialects import pulse, quir


dur_val = 5
amp_r_val = 2.5
amp_i_val = 0.0


with Context(), Location.unknown():
    pulse.pulse.register_dialect()
    quir.quir.register_dialect()

    wf = pulse.WaveformType.get()

    f64 = F64Type.get()
    i32 = IntegerType.get_signless(32)

    i1 = IntegerType.get_signless(1)

    module = Module.create()

    with InsertionPoint(module.body):
        test = pulse.SequenceOp("test", [], [i1])
        test.add_entry_block()

    with InsertionPoint(test.entry_block):
        dur = arith.ConstantOp(i32, dur_val)

        amp_r = arith.ConstantOp(f64, amp_r_val)
        amp_i = arith.ConstantOp(f64, amp_i_val)
        amp = complex.CreateOp(ComplexType.get(F64Type.get()), amp_r, amp_i)

        const = pulse.ConstOp(dur, amp)

        ret = arith.ConstantOp(i1, 0)

        pulse.ReturnOp([ret])

print(module)

# writing to a file
print("Writing to SequenceEx.mlir")
test = open("SequenceEx.mlir", "w")
try:
    test.write(str(module))
except OSError:
    pass
test.close()
print("Done!")
