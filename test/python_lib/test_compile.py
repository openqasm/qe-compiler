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
    Severity,
)
from qss_compiler import exceptions


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


def test_compile_roundtrip_bytecode(example_mlir_str):
    """Test that we can compile a file input via the interface compile_file
    to MLIR"""

    mlir1 = compile_str(
        example_mlir_str,
        input_type=InputType.MLIR,
        output_type=OutputType.MLIR,
    )

    bytecode = compile_str(
        example_mlir_str,
        input_type=InputType.MLIR,
        output_type=OutputType.BYTECODE,
    )

    mlir2 = compile_str(
        bytecode,
        input_type=InputType.BYTECODE,
        output_type=OutputType.MLIR,
    )
    assert mlir2 == mlir1


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


def test_empty_str():
    """Test that we can compile an empty string."""

    _ = compile_str(
        "",
        input_type=InputType.QASM3,
        output_type=OutputType.MLIR,
        output_file=None,
    )


def test_compile_no_output(example_qasm3_str):
    """Test compiling without output."""

    output = compile_str(
        example_qasm3_str,
        input_type=InputType.QASM3,
        output_type=OutputType.NONE,
    )
    assert output is None


def test_compile_invalid_file(example_invalid_qasm3_tmpfile):
    """Test that we can attempt to compile invalid OpenQASM 3 and receive an
    error"""

    with pytest.raises(exceptions.OpenQASM3ParseFailure):
        compile_file(
            example_invalid_qasm3_tmpfile,
            return_diagnostics=True,  # For testing purposes
            input_type=InputType.QASM3,
            output_type=OutputType.MLIR,
            output_file=None,
        )


def test_compile_invalid_str(example_invalid_qasm3_str):
    """Test that we can attempt to compile invalid OpenQASM 3 and receive an
    error"""

    with pytest.raises(exceptions.OpenQASM3ParseFailure) as compfail:
        compile_str(
            example_invalid_qasm3_str,
            return_diagnostics=True,  # For testing purposes
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


def test_warning_not_in_errors(example_warning_not_in_errors):
    """Test that we can attempt to compile invalid OpenQASM 3 and receive an
    error"""

    with pytest.raises(exceptions.OpenQASM3ParseFailure) as compfail:
        compile_str(
            example_warning_not_in_errors,
            return_diagnostics=True,
            input_type=InputType.QASM3,
            output_type=OutputType.MLIR,
            output_file=None,
        )

    assert hasattr(compfail.value, "diagnostics")

    diags = compfail.value.diagnostics

    assert any(
        diag.severity == Severity.Error
        and diag.category == ErrorCategory.OpenQASM3ParseFailure
        and "Non-existent angle a passed as angle argument to Gate Call." in diag.message
        and "^" in diag.message
        for diag in diags
    )
    assert any("OpenQASM 3 parse error" in str(diag) for diag in diags)

    # check string representation of the exception to contain diagnostic messages
    assert (
        "Error: OpenQASM 3 parse error" in str(compfail.value)
        and "Non-existent angle a passed as angle argument to Gate Call." in str(compfail.value)
        and "Angle value exceeds 2pi." not in str(compfail.value)
    )


def test_failure_no_hang():
    """Test no hang on malformed inputs."""
    with pytest.raises(exceptions.QSSCompilerEOFFailure):
        _ = compile_str(
            "",
            input_type=InputType.QASM3,
            output_type=OutputType.MLIR,
            output_file=None,
            extra_args=["bad_arg"],
        )
