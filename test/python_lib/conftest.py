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
Pytest configuration and fixtures
"""

import multiprocessing
import pytest


@pytest.fixture(scope="session", autouse=True)
def set_multiprocessing_start_method():
    multiprocessing.set_start_method("spawn", force=True)


@pytest.fixture
def example_qasm3_str():
    return """OPENQASM 3.0;
    qubit $0;
    bit c0;
    U(1.57079632679, 0.0, 3.14159265359) $0;
    measure $0 -> c0;
    """


@pytest.fixture
def example_mlir_str():
    return """
    func.func @dummy() {
        return
    }
    """


def __create_tmpfile(tmp_path, source):
    tmp_path = tmp_path / "example.qasm"
    with open(tmp_path, "w") as tmpfile:
        tmpfile.write(source)
        tmpfile.flush()
    return tmp_path


@pytest.fixture
def example_qasm3_tmpfile(tmp_path, example_qasm3_str):
    return __create_tmpfile(tmp_path, example_qasm3_str)


@pytest.fixture
def example_invalid_qasm3_str():
    return """OPENQASM 4.0;
    crambit $0;
    fit c0 = ~0;
    """


@pytest.fixture
def example_qasm3_invalid_tmpfile(tmp_path, example_invalid_qasm3_str):
    return __create_tmpfile(tmp_path, example_invalid_qasm3_str)


@pytest.fixture
def example_invalid_qasm3_tmpfile(tmp_path, example_invalid_qasm3_str):
    return __create_tmpfile(tmp_path, example_invalid_qasm3_str)


@pytest.fixture
def example_unsupported_qasm3_str():
    return """OPENQASM 3.0;
    int a;
    int b;
    int c;
    c = a + b;
    """


@pytest.fixture
def example_unsupported_qasm3_tmpfile(tmp_path, example_unsupported_qasm3_str):
    return __create_tmpfile(tmp_path, example_unsupported_qasm3_str)
