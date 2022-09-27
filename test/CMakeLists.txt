# (C) Copyright IBM 2021.
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
