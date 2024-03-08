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

function(qss_dialects_target_includes target)
    set(_DIRS
        ${MLIR_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include
    )
    set_property(TARGET ${target} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
        $<BUILD_INTERFACE:${_DIRS}>)
    if(TARGET obj.${target})
        set_property(TARGET ${target} APPEND PROPERTY INCLUDE_DIRECTORIES
            $<BUILD_INTERFACE:${_DIRS}>)
    endif()
endfunction()


set(GLOBAL APPEND PROPERTY QSSC_CAPI_LIBS)
function(add_qss_upstream_c_api_library name)
    add_mlir_public_c_api_library(${name} ${ARGN})
    set(GLOBAL APPEND PROPERTY QSSC_CAPI_LIBS ${name})
    qss_dialects_target_includes(${name})
endfunction()

add_subdirectory(Dialect)
