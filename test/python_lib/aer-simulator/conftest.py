# (C) Copyright IBM 2023.
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
"""
Pytest configuration and fixtures
"""

import multiprocessing
import pathlib
import pytest
import zipfile
from typing import Iterable


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
def simulator_config_file():
    pytest_dir = pathlib.Path(__file__).parent.resolve()
    simulator_test_cfg = pytest_dir.parent / "aer-simulator/test.cfg"
    assert simulator_test_cfg.exists()
    return str(simulator_test_cfg)
