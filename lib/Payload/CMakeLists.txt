# (C) Copyright IBM 2021.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Register payloads with build system
foreach(payload_dir ${QSSC_PAYLOAD_DIRS})
    message(STATUS "Adding QSS Payload directory: ${payload_dir}")
    add_subdirectory(${payload_dir} ${CMAKE_CURRENT_BINARY_DIR}/Payloads/${payload_dir})
endforeach()

# Write Payloads.inc file for registration
get_property(reg_headers GLOBAL PROPERTY QSSC_PAYLOAD_REGISTRATION_HEADERS)
foreach(REG_HEADER ${reg_headers})
    set(PAYLOAD_REGISTRATION_INCLUDE_SECTION "${PAYLOAD_REGISTRATION_INCLUDE_SECTION}#include \"${REG_HEADER}\"\n")
endforeach()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Payloads.inc.in ${CMAKE_CURRENT_BINARY_DIR}/Payloads.inc)
unset(PAYLOAD_REGISTRATION_INCLUDE_SECTION)

# Add QSSCPayload lib
get_property(qssc_payloads GLOBAL PROPERTY QSSC_PAYLOADS)
qssc_add_library(QSSCPayload
        PayloadRegistry.cpp
        Payload.cpp

        ADDITIONAL_HEADER_DIRS
        ${QSSC_INCLUDE_DIR}/Payload

        LINK_LIBS
        ${qssc_payloads}
        libzip::libzip
        )

# Add include directory to pick up generated Payloads.inc
target_include_directories(QSSCPayload PRIVATE ${QSSC_BINARY_DIR}/lib/Payload/)
