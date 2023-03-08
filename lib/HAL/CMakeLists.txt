# (C) Copyright IBM 2023.
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
    TargetRegistry.cpp
    TargetSystem.cpp

    ADDITIONAL_HEADER_DIRS
    ${QSSC_INCLUDE_DIR}/HAL

    LINK_LIBS
    ${qssc_targets}
)

# Add include directory to pick up generated Targets.inc
target_include_directories(QSSCHAL PRIVATE ${QSSC_BINARY_DIR}/lib/HAL/)
