function(qss_dialects_target_includes target)
    set(_DIRS
        ${MLIR_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_BINARY_DIR}/include
    )
    set_property(TARGET ${target} APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES
        $<BUILD_INTERFACE:${_DIRS}>)
    if(TARGET obj.${target})
        set_property(TARGET ${target} APPEND PROPERTY INCLUDE_DIRECTORIES
            $<BUILD_INTERFACE:${_DIRS}>)
    endif()
endfunction()


set(GLOBAL APPEND PROPERTY QSSC_CAPI_LIBS)
function(add_qss_upstream_c_api_library name)
    add_mlir_public_c_api_library(${name} ${ARGN})
    set(GLOBAL APPEND PROPERTY QSSC_CAPI_LIBS ${name})
    qss_dialects_target_includes(${name})
    # add to include dirs
    # set(_DIRS
    #     ${MLIR_INCLUDE_DIRS}
    #     ${PROJEC}
    # )
endfunction()

add_subdirectory(Dialect)

# Build the CAPI dylib.
# if(MLIR_BUILD_MLIR_C_DYLIB)
#   message(STATUS "Building MLIR-C dylib")
#   get_property(_capi_libraries GLOBAL PROPERTY QSS_CAPI_LIBRARIES)
#   add_mlir_aggregate(QSS-C
#     SHARED
#     EMBED_LIBS
#       ${_capi_libraries}
#   )
#   if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
#     target_link_options(MLIR-C PRIVATE "-Wl,-exclude-libs,ALL")
#   else()
#     if(NOT CMAKE_C_VISIBILITY_PRESET STREQUAL "hidden" OR NOT CMAKE_CXX_VISIBILITY_PRESET STREQUAL "hidden")
#       message(STATUS "MLIR-C on this platform exports all symbols. Recommend building with CMAKE_(C|CXX)_VISIBILITY_PRESET=hidden or implement filtering support.")
#     endif()
#   endif()
# endif()