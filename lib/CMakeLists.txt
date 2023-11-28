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

add_subdirectory(API)
add_subdirectory(Config)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Frontend)
add_subdirectory(HAL)
add_subdirectory(Arguments)
add_subdirectory(Payload)
add_subdirectory(Utils)

set(SOURCES
        QSSC.cpp
)
add_library(QSSCCore ${SOURCES})
add_dependencies(QSSCCore mlir-headers)

get_property(qssc_targets GLOBAL PROPERTY QSSC_TARGETS)
get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(mlir_extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

# All libraries for the QSS-Compiler
set(qssc_api_libs
        ${qssc_targets}
        QSSCAPI
        QSSCHAL
        QSSCOpenQASM3Frontend
        QSSConfig
        QSSCArguments
        QSSCPayload
        QSSCCore
        ${mlir_dialect_libs}
        ${mlir_conversion_libs}
        ${mlir_extension_libs}
        MLIRCallInterfaces
        MLIRExecutionEngine
        MLIROptLib
        MLIRArithDialect
        MLIRLLVMDialect
        MLIROQ3Dialect
        MLIRQCSDialect
        MLIRQUIRDialect
        MLIRPulseDialect
        MLIRSCFDialect
        qasm::qasm
        mpc::mpc
        mpfr::mpfr
        gmp::gmp
        )
set_property(GLOBAL PROPERTY QSSC_API_LIBS ${qssc_api_libs})

# collect all required components in an "interface library", which exports its
# INTERFACE and dependencies to all targets that refer to the library yet does
# not materialize as an artifact by itself
add_library(QSSCLib INTERFACE)
add_dependencies(QSSCLib mlir-headers)
target_link_libraries(QSSCLib INTERFACE ${qssc_api_libs})

if(INSTALL_BUNDLED_STATIC_LIB)
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        message(FATAL_ERROR "Bundling qss-compiler into a single static library on MacOS is not yet supported.")
        # If you consider adding that support, MacOS's libtool could be an option since it appears capable of bundling static libraries.
    endif()
    add_library(QSSCompiler STATIC API/api.cpp)
    add_dependencies(QSSCompiler mlir-headers)
    target_link_libraries(QSSCompiler PUBLIC
           ${qssc_targets}
            QSSCAPI
            QSSCHAL
            QSSCOpenQASM3Frontend
            QSSConfig
            QSSCArguments
            QSSCPayload
            QSSCPayloadZip
            QSSCCore
            QSSCUtils
            ${mlir_dialect_libs}
            ${mlir_conversion_libs}
    )
    bundle_static_library(QSSCompiler QSSCompilerBundled)
    install(FILES
        ${CMAKE_BINARY_DIR}/libQSSCompilerBundled.a
        DESTINATION
        "${CMAKE_INSTALL_LIBDIR}"
    )
    # install API library and its headers
    install(TARGETS
            QSSCAPI
        LIBRARY DESTINATION
            "${CMAKE_INSTALL_LIBDIR}"
        FILE_SET HEADERS DESTINATION
            "${CMAKE_INSTALL_INCLUDEDIR}/API"

    )

endif()
