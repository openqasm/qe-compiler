# add_library(MLIRCAPIPulse Pulse.cpp)
# add_library(MLIRCAPIQUIR QUIR.cpp)
# add_library(MLIRCAPIQCS QCS.cpp)
# add_library(MLIRCAPIOQ3 OQ3.cpp)
add_qss_upstream_c_api_library(MLIRCAPIPulse
    Pulse.cpp

    PARTIAL_SOURCES_INTENDED
    LINK_LIBS PUBLIC
    MLIRCAPIIR
    MLIRPulseDialect
    MLIRPulseTransforms
)

add_qss_upstream_c_api_library(MLIRCAPIQUIR
    QUIR.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/QUIR

    PARTIAL_SOURCES_INTENDED
    LINK_LIBS PUBLIC
    MLIRCAPIIR
    MLIRQUIRDialect
    MLIRQUIRTransforms
    MLIRQUIRUtils
)

add_qss_upstream_c_api_library(MLIRCAPIQCS
    QCS.cpp

    PARTIAL_SOURCES_INTENDED
    LINK_LIBS PUBLIC
    MLIRCAPIIR
    MLIRQCSDialect
)

add_qss_upstream_c_api_library(MLIRCAPIOQ3
    OQ3.cpp

    PARTIAL_SOURCES_INTENDED
    LINK_LIBS PUBLIC
    MLIRCAPIIR
    MLIROQ3Dialect
    MLIROQ3Transforms
)

set(mlir_qssc_capi_libs
    MLIRCAPIPulse
    MLIRCAPIQUIR
    MLIRCAPIQCS
    MLIRCAPIOQ3
    CACHE STRING
    "List of capi libs"
    FORCE
)
