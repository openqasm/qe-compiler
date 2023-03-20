# (C) Copyright IBM 2021, 2023.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

set(QSSC_PAYLOAD_DIRS
        ${QSSC_PAYLOAD_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        )

qssc_add_plugin(QSSCPayloadZip QSSC_PAYLOAD_PLUGIN
        ZipPayload.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${QSSC_INCLUDE_DIR}/Payload

        LINK_LIBS
        QSSCPayload
        libzip::zip

        PLUGIN_REGISTRATION_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/Payload.inc
        )
