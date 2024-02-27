# (C) Copyright IBM 2024.
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

add_qss_upstream_c_api_library(MLIRCAPIPulse
    Pulse.cpp

    PARTIAL_SOURCES_INTENDED
    LINK_LIBS PUBLIC
    MLIRCAPIIR
    MLIRPulseDialect
    MLIRPulseTransforms
)

add_qss_upstream_c_api_library(MLIRCAPIQUIR
    QUIR.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/QUIR

    PARTIAL_SOURCES_INTENDED
    LINK_LIBS PUBLIC
    MLIRCAPIIR
    MLIRQUIRDialect
    MLIRQUIRTransforms
    MLIRQUIRUtils
)

add_qss_upstream_c_api_library(MLIRCAPIQCS
    QCS.cpp

    PARTIAL_SOURCES_INTENDED
    LINK_LIBS PUBLIC
    MLIRCAPIIR
    MLIRQCSDialect
)

add_qss_upstream_c_api_library(MLIRCAPIOQ3
    OQ3.cpp

    PARTIAL_SOURCES_INTENDED
    LINK_LIBS PUBLIC
    MLIRCAPIIR
    MLIROQ3Dialect
    MLIROQ3Transforms
)

set(mlir_qssc_capi_libs
    MLIRCAPIPulse
    MLIRCAPIQUIR
    MLIRCAPIQCS
    MLIRCAPIOQ3
    CACHE STRING
    "List of capi libs"
    FORCE
)
