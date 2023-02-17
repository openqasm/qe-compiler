# (C) Copyright IBM 2023.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
find_package(CapnProto CONFIG REQUIRED)
include_directories(
        ${CapnProto_INCLUDE_DIRS}        
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}/generated)

find_file(SCHEMA_FILE mock.capnp
          PATHS ${PROJECT_SOURCE_DIR}/mock_target/schema)
get_filename_component(CAPNPC_SRC_PREFIX ${SCHEMA_FILE} DIRECTORY)
set(CAPNPC_OUTPUT_DIR ${CMAKE_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${CAPNPC_OUTPUT_DIR})
capnp_generate_cpp(QSSC_PAYLOADV2_SOURCES QSSC_PAYLOADV2_HEADERS ${SCHEMA_FILE})

set(SOURCES
        PayloadV2.cpp
        ${QSSC_PAYLOADV2_SOURCES})

set(LIBRARY_NAME QSSCPayloadV2)
add_library(${LIBRARY_NAME} ${SOURCES})
target_link_libraries(${LIBRARY_NAME} PRIVATE CapnProto::capnp)
