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

enable_testing()

# Configure core unit testing.
add_subdirectory(unittest)

# Target for all LIT testing.
add_custom_target(check-qss-compiler COMMENT "Running LIT suites")

# Target for all testing.
add_custom_target(check-tests DEPENDS run-qss-compiler-unittests check-qss-compiler COMMENT "Running all tests")

# Add QSS core LIT suite.
qssc_add_lit_test_suite(check-qss-core "QSS Compiler Core" "${CMAKE_CURRENT_SOURCE_DIR}" "")

# Add target LIT suites.
foreach(lit_dir ${QSSC_TARGET_TEST_DIRS})
    message(STATUS "Adding LIT directory: ${lit_dir}")
    add_subdirectory("${lit_dir}" "${CMAKE_CURRENT_BINARY_DIR}/Targets/${lit_dir}")
endforeach()
