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

set(QSSC_PAYLOAD_PATHS
        ${QSSC_PAYLOAD_PATHS}
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
