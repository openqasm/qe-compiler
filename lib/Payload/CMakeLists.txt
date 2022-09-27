# (C) Copyright IBM 2021.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

ADD_LIBRARY(QSSCPayload Payload.cpp)
include_directories(${PROJECT_SOURCE_DIR}/include)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
set(LIBS
        libzip::libzip
        )

target_link_libraries(QSSCPayload PRIVATE ${LIBS})
