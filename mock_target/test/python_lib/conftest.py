# (C) Copyright IBM 2022.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
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
def example_invalid_qasm3_str():
    return """FOOBAR 3.0;
    crambit $0;
    fit c0 = ~0;
    """


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


@pytest.fixture
def mock_config_file():
    pytest_dir = pathlib.Path(__file__).parent.resolve()
    mock_test_cfg = pytest_dir.parent / "test.cfg"
    assert mock_test_cfg.exists()
    return str(mock_test_cfg)


@pytest.fixture
def check_payload():
    def check_payload_(payload_filelike, expected_files: Iterable[str] = ()):
        zf = zipfile.ZipFile(payload_filelike, "r")

        # check checksums in zip file (i.e., no corruption)
        first_bad_file = zf.testzip()
        assert first_bad_file is None, "found corrupted file in payload: " + str(first_bad_file)

        # check that payload contains manifest
        assert "manifest/manifest.json" in zf.namelist()

        for expected_file in expected_files:
            assert expected_file in zf.namelist()

    return check_payload_
