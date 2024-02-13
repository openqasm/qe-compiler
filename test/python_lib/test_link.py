# (C) Copyright IBM 2023.
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
Unit tests for the linker API.
"""
import pytest

from qss_compiler import link_file
from qss_compiler.exceptions import QSSLinkerNotImplemented


def test_linker_not_implemented(tmp_path):
    qem_file = tmp_path / "test.txt"
    arg_file = tmp_path / "dummy.txt"
    with open(qem_file, "w") as f:
        f.write("dummy")

    with pytest.raises(QSSLinkerNotImplemented) as error:
        link_file(
            input_file=qem_file,
            output_file=arg_file,
            target="Mock",
            arguments={},
        )

    assert str(error.value.message) == "Unable to load bind arguments implementation for target."
