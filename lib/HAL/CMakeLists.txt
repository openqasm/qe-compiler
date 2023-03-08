# (C) Copyright IBM 2021, 2023.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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
)

# Add include directory to pick up generated Targets.inc
target_include_directories(QSSCHAL PRIVATE ${QSSC_BINARY_DIR}/lib/HAL/)
