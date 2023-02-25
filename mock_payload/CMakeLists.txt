# (C) Copyright IBM 2021, 2023.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

qssc_add_system_payload(QSSCPayloadMock
        MockPayload.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${QSSC_INCLUDE_DIR}/Payload

        LINK_LIBS
        QSSCPayload

        PAYLOAD_REGISTRATION_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/Payload.inc
        )
