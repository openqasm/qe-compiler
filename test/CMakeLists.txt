# (C) Copyright IBM 2021, 2023.
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
