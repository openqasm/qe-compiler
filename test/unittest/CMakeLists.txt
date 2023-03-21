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

package_add_test_with_libs(unittest-quir-dialect
        quir-dialect.cpp

        LIBRARIES
        QSSCLib
)

set(TEST_FILES
        Payload/PayloadRegistryTest.cpp
        # add other source files with test cases here
        )

if (QSSC_WITH_MOCK_TARGET)
    set(TEST_FILES
            HAL/TargetSystemRegistryTest.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}
            )
endif ()

package_add_test_with_libs(unittest-qss-compiler
        ${TEST_FILES}

        LIBRARIES
        QSSCLib
        )

add_custom_target(run-qss-compiler-unittests
        COMMAND ${CMAKE_CTEST_COMMAND}
        DEPENDS ${QSSC_UNITTESTS}
        )
