# (C) Copyright IBM 2023.
#
# This code is part of Qiskit.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unit tests for the compiler API.
"""
import pytest
import qss_compiler
from qss_compiler import (
    compile_file,
    compile_str,
    ErrorCategory,
    InputType,
    OutputType,
    QSSCompilationFailure,
    Severity,
)


def check_mlir_string(mlir):
    assert isinstance(mlir, str)
    assert "module" in mlir
    assert "qcs.init" in mlir


def test_attributes():
    assert qss_compiler.__doc__ == "Python bindings for the QSS Compiler."


def test_compile_file_to_mlir(example_qasm3_tmpfile):
    """Test that we can compile a file input via the interface compile_file
    to MLIR"""

    mlir = compile_file(
        example_qasm3_tmpfile,
        input_type=InputType.QASM3,
        output_type=OutputType.MLIR,
        output_file=None,
    )
    # with output type MLIR, expect string decoding in python wrapper
    check_mlir_string(mlir)


def test_compile_file_to_mlir_idempotence(example_qasm3_tmpfile):
    """Test that we can compile a file input via the interface compile_file
    to MLIR"""

    mlir = compile_file(
        example_qasm3_tmpfile,
        input_type=InputType.QASM3,
        output_type=OutputType.MLIR,
        output_file=None,
    )
    # with output type MLIR, expect string decoding in python wrapper
    check_mlir_string(mlir)

    mlir2 = compile_file(
        example_qasm3_tmpfile,
        input_type=InputType.QASM3,
        output_type=OutputType.MLIR,
        output_file=None,
    )
    assert mlir2 == mlir


def test_compile_str_to_mlir(example_qasm3_str):
    """Test that we can compile a string input via the interface
    compile_str to an MLIR output"""

    mlir = compile_str(
        example_qasm3_str,
        input_type=InputType.QASM3,
        output_type=OutputType.MLIR,
        output_file=None,
    )
    check_mlir_string(mlir)


def test_compile_invalid_file(example_invalid_qasm3_tmpfile):
    """Test that we can attempt to compile invalid OpenQASM 3 and receive an
    error"""

    with pytest.raises(QSSCompilationFailure):
        compile_file(
            example_invalid_qasm3_tmpfile,
            input_type=InputType.QASM3,
            output_type=OutputType.MLIR,
            output_file=None,
        )


def test_compile_invalid_str(example_invalid_qasm3_str):
    """Test that we can attempt to compile invalid OpenQASM 3 and receive an
    error"""

    with pytest.raises(QSSCompilationFailure) as compfail:
        compile_str(
            example_invalid_qasm3_str,
            input_type=InputType.QASM3,
            output_type=OutputType.MLIR,
            output_file=None,
        )

    assert hasattr(compfail.value, "diagnostics")

    diags = compfail.value.diagnostics
    assert any(
        diag.severity == Severity.Error
        and diag.category == ErrorCategory.OpenQASM3ParseFailure
        and "unknown version number" in diag.message
        and "^" in diag.message
        for diag in diags
    )
    assert any("OpenQASM 3 parse error" in str(diag) for diag in diags)

    # check string representation of the exception to contain diagnostic messages
    assert "OpenQASM 3 parse error" in str(compfail.value)
    assert "unknown version number" in str(compfail.value)
