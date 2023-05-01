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

add_subdirectory(API)
add_subdirectory(Config)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Frontend)
add_subdirectory(HAL)
add_subdirectory(Payload)
add_subdirectory(Utils)

set(SOURCES
        QSSC.cpp
)
add_library(QSSCCore ${SOURCES})

get_property(qssc_targets GLOBAL PROPERTY QSSC_TARGETS)
get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# All liibraries for the QSS-Compiler
set(qssc_api_libs
        ${qssc_targets}
        QSSCAPI
        QSSCHAL
        QSSCOpenQASM3Frontend
        QSSConfig
        QSSCPayload
        QSSCCore
        ${mlir_dialect_libs}
        ${mlir_conversion_libs}
        MLIRCallInterfaces
        MLIRExecutionEngine
        MLIROptLib
        MLIROQ3Dialect
        MLIRQCSDialect
        MLIRQUIRDialect
        MLIRPulseDialect
        qasm::qasm
        mpc::mpc
        mpfr::mpfr
        gmp::gmp
        )
set_property(GLOBAL PROPERTY QSSC_API_LIBS ${qssc_api_libs})

# collect all required components in an "interface library", which exports its
# INTERFACE and dependencies to all targets that refer to the library yet does
# not materialize as an artifact by itself
add_library(QSSCLib INTERFACE)
target_link_libraries(QSSCLib INTERFACE ${qssc_api_libs})

# Expose (i.e., write to disk) static library with same contents as `QSSCLib`,
# including interface to `compile()` in `api.h`.
qssc_add_library(QSSC "API/api.cpp" ADDITIONAL_HEADER_DIRS ${QSSC_INCLUDE_DIR}/API)
target_link_libraries(QSSC ${qssc_api_libs})
