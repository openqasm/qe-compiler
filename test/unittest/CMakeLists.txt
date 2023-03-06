# (C) Copyright IBM 2022, 2023.
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

package_add_test_with_libs(unittest-quir-dialect
        quir-dialect.cpp

        LIBRARIES
        QSSCLib
)

if (QSSC_WITH_MOCK_TARGET)
    package_add_test_with_libs(unittest-qss-compiler
            HAL/TargetRegistryTest.cpp
            # add other source files with test cases here

            LIBRARIES
            QSSCLib
            )
else()
    message(WARNING "Unit testing is enabled, but QSSC_WITH_MOCK_TARGET is false. QSS compiler unit tests will be skipped.")
endif ()

add_custom_target(run-qss-compiler-unittests
        COMMAND ${CMAKE_CTEST_COMMAND}
        DEPENDS ${QSSC_UNITTESTS}
        )
