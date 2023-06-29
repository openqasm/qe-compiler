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

add_library(QSSCAPI api.cpp error.cpp)

set(LIBS
        qasm::qasm
        )
target_link_libraries(QSSCAPI ${LIBS})

target_sources(QSSCAPI
    PRIVATE api.cpp error.cpp
    INTERFACE FILE_SET HEADERS
    BASE_DIRS ${QSSC_INCLUDE_DIR}/API
    FILES ${QSSC_INCLUDE_DIR}/API/api.h ${QSSC_INCLUDE_DIR}/API/error.h
    )

add_dependencies(QSSCAPI MLIROQ3Dialect MLIRQCSDialect MLIRQUIRDialect)
