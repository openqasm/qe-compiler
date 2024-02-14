### A simple example of creating an SequenceOp using the pulse python bindings.


from qss_compiler.mlir.ir import *
# from mlir import affine
# from mlir.ir import F64Type, IntegerType, IndexType, ComplexType
from qss_compiler.mlir.dialects import arith, builtin, std, scf, linalg, complex

from qss_compiler.mlir.dialects import pulse, quir # noqa: F401, E402


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
        amp = complex.CreateOp(amp_r, amp_i)

        const = pulse.ConstOp(wf, dur, amp)

        ret = arith.ConstantOp(i1, 0)

        pulse.ReturnOp([ret])

print(module)

# writing to a file
print("Writing to test.mlir")
test = open("SequenceEx.mlir", "w")
try:
    test.write(str(module))
except:
    test.write(str(module))
test.close()
print("Done!")
