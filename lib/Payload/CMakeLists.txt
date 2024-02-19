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

add_subdirectory(ZipPayload)

# Register payloads with build system
foreach(payload_path ${QSSC_PAYLOAD_PATHS})
    message(STATUS "Adding QSS Payload directory: ${payload_path}")
    add_subdirectory(${payload_path} ${CMAKE_CURRENT_BINARY_DIR}/Payloads/${payload_path})
endforeach()

# Write Payloads.inc file for registration
get_property(reg_headers GLOBAL PROPERTY QSSC_PAYLOAD_REGISTRATION_HEADERS)
foreach(REG_HEADER ${reg_headers})
    set(PAYLOAD_REGISTRATION_INCLUDE_SECTION "${PAYLOAD_REGISTRATION_INCLUDE_SECTION}#include \"${REG_HEADER}\"\n")
endforeach()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Payloads.inc.in ${CMAKE_CURRENT_BINARY_DIR}/Payloads.inc)
unset(PAYLOAD_REGISTRATION_INCLUDE_SECTION)

# Add QSSCHAL lib
get_property(qssc_payloads GLOBAL PROPERTY QSSC_PAYLOADS)
qssc_add_library(QSSCPayload
        Payload.cpp

        ADDITIONAL_HEADER_DIRS
        ${QSSC_INCLUDE_DIR}/Payload

        LINK_LIBS
        ${qssc_payloads}
        QSSCArguments
        QSSCError
)

# Add include directory to pick up generated Payloads.inc
target_include_directories(QSSCPayload PRIVATE ${QSSC_BINARY_DIR}/lib/Payload/)
