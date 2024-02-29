# (C) Copyright IBM 2023, 2024.
#
# This code is part of Qiskit.
#
# This code is licensed under the Apache License, Version 2.0 with LLVM
# Exceptions. You may obtain a copy of this license in the LICENSE.txt
# file in the root directory of this source tree.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Unit tests for the dialect python bindings
"""
import pytest  # noqa: F401

from qss_compiler.mlir.ir import *  # noqa: F401, F403
from qss_compiler.mlir.ir import (
    Context,
    InsertionPoint,
    Module,
    Location,
)
from qss_compiler.mlir.dialects import (  # noqa: F401
    arith,
    builtin,
    pulse,
    qcs,
    func,
)

from test_compile import check_mlir_string


def test_create_sequenceop():
    with Context(), Location.unknown():
        pulse.pulse.register_dialect()
        qcs.qcs.register_dialect()
        module = Module.create()

        with InsertionPoint(module.body):
            seq = pulse.SequenceOp("test_seq", [], [])
            seq.add_entry_block()

            function = func.FuncOp("test_main", ([], []))
            function.add_entry_block()

        with InsertionPoint(seq.entry_block):
            pulse.ReturnOp([])

        with InsertionPoint(function.entry_block):
            qcs.SystemInitOp()
            res = pulse.CallSequenceOp([], "test_seq", [])
            func.ReturnOp(res)
            qcs.SystemFinalizeOp

    check_mlir_string(str(module))
