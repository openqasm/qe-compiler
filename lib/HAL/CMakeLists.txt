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

add_subdirectory(Compile)

# Register targets with build system
foreach(target_dir ${QSSC_TARGET_DIRS})
    message(STATUS "Adding QSS Target directory: ${target_dir}")
    add_subdirectory(${target_dir} ${CMAKE_CURRENT_BINARY_DIR}/Targets/${target_dir})
endforeach()

# Write Targets.inc file for registration
get_property(reg_headers GLOBAL PROPERTY QSSC_TARGET_REGISTRATION_HEADERS)
foreach(REG_HEADER ${reg_headers})
    set(TARGET_REGISTRATION_INCLUDE_SECTION "${TARGET_REGISTRATION_INCLUDE_SECTION}#include \"${REG_HEADER}\"\n")
endforeach()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Targets.inc.in ${CMAKE_CURRENT_BINARY_DIR}/Targets.inc)
unset(TARGET_REGISTRATION_INCLUDE_SECTION)

# Add QSSCHAL lib
get_property(qssc_targets GLOBAL PROPERTY QSSC_TARGETS)
qssc_add_library(QSSCHAL
    PassRegistration.cpp
    SystemConfiguration.cpp
    TargetSystem.cpp
    TargetSystemInfo.cpp
    TargetSystemRegistry.cpp

    ADDITIONAL_HEADER_DIRS
    ${QSSC_INCLUDE_DIR}/HAL

    LINK_LIBS
    ${qssc_targets}
    QSSCArguments
    QSSCHALCompile

)

# Add include directory to pick up generated Targets.inc
target_include_directories(QSSCHAL PRIVATE ${QSSC_BINARY_DIR}/lib/HAL/)
