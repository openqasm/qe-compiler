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
Unit tests for the compiler API using mock targets.
"""

import asyncio
from datetime import datetime, timedelta
import io
import os
import pytest

from qss_compiler import (
    compile_file,
    compile_file_async,
    compile_str,
    compile_str_async,
    InputType,
    OutputType,
    CompileOptions,
)
from qss_compiler.exceptions import QSSCompilationFailure

compiler_extra_args = ["--enable-circuits=false"]


def check_mlir_string(mlir):
    assert isinstance(mlir, str)
    assert "module" in mlir
    assert "qcs.init" in mlir


def test_compile_file_to_qem(example_qasm3_tmpfile, mock_config_file, check_payload):
    """Test that we can compile a file input via the interface compile_file
    to a QEM payload"""

    qem = compile_file(
        example_qasm3_tmpfile,
        input_type=InputType.QASM3,
        output_type=OutputType.QEM,
        output_file=None,
        target="mock",
        config_path=mock_config_file,
        extra_args=compiler_extra_args,
    )
    # QEM payload is returned as byte sequence
    assert isinstance(qem, bytes)

    # check that payload is a zip file
    payload_filelike = io.BytesIO(qem)
    check_payload(payload_filelike)


def test_compile_str_to_qem(mock_config_file, example_qasm3_str, check_payload):
    """Test that we can compile an OpenQASM3 string via the interface
    compile_file to a QEM payload"""

    qem = compile_str(
        example_qasm3_str,
        input_type=InputType.QASM3,
        output_type=OutputType.QEM,
        output_file=None,
        target="mock",
        config_path=mock_config_file,
        extra_args=compiler_extra_args,
    )
    # QEM payload is returned as byte sequence
    assert isinstance(qem, bytes)

    # check that payload is a zip file
    payload_filelike = io.BytesIO(qem)
    check_payload(payload_filelike)


def test_compile_file_to_qem_file(example_qasm3_tmpfile, mock_config_file, tmp_path, check_payload):
    """Test that we can compile a file input via the interface compile_file
    to a QEM payload into a file"""
    tmpfile = tmp_path / "payload.qem"

    result = compile_file(
        example_qasm3_tmpfile,
        input_type=InputType.QASM3,
        output_type=OutputType.QEM,
        output_file=tmpfile,
        target="mock",
        config_path=mock_config_file,
        extra_args=compiler_extra_args,
    )

    # no direct return
    assert result is None
    file_stat = os.stat(tmpfile)
    assert file_stat.st_size > 0

    with open(tmpfile, "rb") as payload:
        check_payload(payload)


def test_compile_str_to_qem_file(mock_config_file, tmp_path, example_qasm3_str, check_payload):
    """Test that we can compile an OpenQASM3 string via the interface
    compile_file to a QEM payload in an output file"""
    tmpfile = tmp_path / "payload.qem"

    result = compile_str(
        example_qasm3_str,
        input_type=InputType.QASM3,
        output_type=OutputType.QEM,
        output_file=tmpfile,
        target="mock",
        config_path=mock_config_file,
        extra_args=compiler_extra_args,
    )

    # no direct return
    assert result is None
    file_stat = os.stat(tmpfile)
    assert file_stat.st_size > 0

    with open(tmpfile, "rb") as payload:
        check_payload(payload)


def test_compile_failing_str_to_qem(
    mock_config_file, example_unsupported_qasm3_str, example_qasm3_str, check_payload
):
    """Test that compiling an invalid OpenQASM3 string via the interface
    compile_str to a QEM payload will fail and result in an empty payload."""

    with pytest.raises(QSSCompilationFailure):
        compile_str(
            example_unsupported_qasm3_str,
            input_type=InputType.QASM3,
            output_type=OutputType.QEM,
            output_file=None,
            target="mock",
            config_path=mock_config_file,
            extra_args=compiler_extra_args,
        )


def test_compile_failing_file_to_qem(
    example_unsupported_qasm3_tmpfile, mock_config_file, tmp_path, check_payload
):
    """Test that compiling an invalid file input via the interface compile_file
    to a QEM payload will fail and result in an empty payload."""

    with pytest.raises(QSSCompilationFailure):
        compile_file(
            example_unsupported_qasm3_tmpfile,
            input_type=InputType.QASM3,
            output_type=OutputType.QEM,
            output_file=None,
            target="mock",
            config_path=mock_config_file,
            extra_args=compiler_extra_args,
        )


def test_compile_options(mock_config_file, example_qasm3_str):
    """Test compilation with explicit CompileOptions construction."""

    compile_options = CompileOptions(
        input_type=InputType.QASM3,
        output_type=OutputType.MLIR,
        target="mock",
        config_path=mock_config_file,
        shot_delay=100,
        num_shots=10000,
        extra_args=compiler_extra_args + ["--mlir-pass-statistics"],
    )

    mlir = compile_str(example_qasm3_str, compile_options=compile_options)
    check_mlir_string(mlir)


async def sleep_a_little():
    await asyncio.sleep(1)
    return datetime.now()


@pytest.mark.asyncio
async def test_async_compile_str(mock_config_file, example_qasm3_str, check_payload):
    """Test that async wrapper produces correct output and does not block the even loop."""
    async_compile = compile_str_async(
        example_qasm3_str,
        input_type=InputType.QASM3,
        output_type=OutputType.QEM,
        output_file=None,
        target="mock",
        config_path=mock_config_file,
        extra_args=compiler_extra_args,
    )
    # Start a task that sleeps shorter than the compilation and then takes a
    # timestamp. If the compilation blocks the event loop, then the timestamp
    # will be delayed further than the intended sleep duration.
    sleeper = asyncio.create_task(sleep_a_little())
    timestamp_launched = datetime.now()
    qem = await async_compile
    timestamp_sleeped = await sleeper

    sleep_duration = timestamp_sleeped - timestamp_launched
    milliseconds_waited = sleep_duration / timedelta(microseconds=1000)
    assert (
        milliseconds_waited <= 1100
    ), f"sleep took longer than intended ({milliseconds_waited} ms instead of ~1000), \
        event loop probably got blocked!"

    # QEM payload is returned as byte sequence
    assert isinstance(qem, bytes)

    # check that payload is a zip file
    payload_filelike = io.BytesIO(qem)
    check_payload(payload_filelike)


@pytest.mark.asyncio
async def test_async_compile_file(example_qasm3_tmpfile, mock_config_file, check_payload):
    """Test that async wrapper produces correct output and does not block the even loop."""
    async_compile = compile_file_async(
        example_qasm3_tmpfile,
        input_type=InputType.QASM3,
        output_type=OutputType.QEM,
        output_file=None,
        target="mock",
        config_path=mock_config_file,
        extra_args=compiler_extra_args,
    )
    # Start a task that sleeps shorter than the compilation and then takes a
    # timestamp. If the compilation blocks the event loop, then the timestamp
    # will be delayed further than the intended sleep duration.
    sleeper = asyncio.create_task(sleep_a_little())
    timestamp_launched = datetime.now()
    qem = await async_compile
    timestamp_sleeped = await sleeper

    sleep_duration = timestamp_sleeped - timestamp_launched
    milliseconds_waited = sleep_duration / timedelta(microseconds=1000)
    assert (
        milliseconds_waited <= 1100
    ), f"sleep took longer than intended ({milliseconds_waited} ms instead of ~1000), \
         event loop probably got blocked!"

    # QEM payload is returned as byte sequence
    assert isinstance(qem, bytes)

    # check that payload is a zip file
    payload_filelike = io.BytesIO(qem)
    check_payload(payload_filelike)
