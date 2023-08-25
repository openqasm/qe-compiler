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

include(AddMLIRPython)

project(py_qssc)

add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=qss_compiler.mlir.")

macro(python_pkg_add_file file lib)
    set(src_file ${CMAKE_CURRENT_SOURCE_DIR}/${file})
    set(dest_file ${CMAKE_CURRENT_BINARY_DIR}/${file})
    add_custom_command(OUTPUT ${dest_file}
      COMMAND ${CMAKE_COMMAND} -E copy
        ${src_file}
        ${dest_file}
      DEPENDS ${src_file}
    )
    LIST(APPEND ${lib} ${dest_file})
endmacro()

# collect python package files from this directory
# into a variable PY_LIB_FILES
macro(python_pkg_add_files)
    foreach(file ${ARGN})
        python_pkg_add_file(${file} PY_LIB_FILES)
    endforeach()
endmacro()

# Copy over python source files and configuration files for setuptools into
# build directory so that the python package ("wheel") can be built there.
# Note that when adding new source files, you need to add them here.
python_pkg_add_files(
                MANIFEST.in
                README.md
                dummy.c
                setup.cfg
                setup.py
                setup_mlir.sh
)

configure_file(pyproject.toml.in ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml)

add_subdirectory(qss_compiler)
add_custom_target(copy_python_files DEPENDS ${PY_LIB_FILES})
add_dependencies(py_qssc copy_python_files)

# TODO: redeclare built-in MLIR python sources and extensions to remove source file dependencies
get_filename_component(LLVM_CMAKE ${LLVM_DIR} DIRECTORY)
get_filename_component(LLVM_LIB ${LLVM_CMAKE} DIRECTORY)
set(LLVM_SRC_PYTHON ${LLVM_LIB} "/src/python/")

declare_mlir_python_sources(QSSPythonSources)
declare_mlir_python_sources(QSSPythonSources.Dialects
  ADD_TO_PARENT QSSPythonSources)


declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT QSSPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/qss_compiler/mlir"
  TD_FILE dialects/QUIR.td
  SOURCES
    dialects/quir.py
    dialects/quir.pyi
    dialects/_quir_ops_ext.py
  DIALECT_NAME quir)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT QSSPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/qss_compiler/mlir"
  TD_FILE dialects/Pulse.td
  SOURCES
    dialects/pulse.py
    dialects/pulse.pyi
    dialects/_pulse_ops_ext.py
  DIALECT_NAME pulse)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT QSSPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/qss_compiler/mlir"
  TD_FILE dialects/OQ3.td
  SOURCES
    dialects/oq3.py
    dialects/OQ3.pyi
    dialects/_oq3_ops_ext.py
  DIALECT_NAME oq3)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT QSSPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/qss_compiler/mlir"
  TD_FILE dialects/QCS.td
  SOURCES
    dialects/qcs.py
    dialects/QCS.pyi
    dialects/_qcs_ops_ext.py
  DIALECT_NAME qcs)

# LLVM 14 does not appear to export the complex dialect
# this may be removed when updating for LLVM 15
declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT QSSPythonSources.Dialects
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/qss_compiler/mlir"
  TD_FILE dialects/ComplexOps.td
  SOURCES
    dialects/complex.py
    dialects/_complex_ops_ext.py
  DIALECT_NAME complex)


set(PYTHON_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib/Bindings/Python/")

declare_mlir_python_extension(QSSPythonExtension.Dialects.Pulse.Pybind
  MODULE_NAME _ibmDialectsPulse
  ADD_TO_PARENT QSSPythonSources.Dialects.pulse
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectPulse.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
    MLIRPulseDialect
    MLIRPulseUtils
    MLIRQUIRDialect
    MLIRQUIRUtils
    MLIROQ3Dialect
    MLIRQCSDialect
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIPulse
)

declare_mlir_python_extension(QSSPythonExtension.Dialects.QUIR.Pybind
  MODULE_NAME _ibmDialectsQUIR
  ADD_TO_PARENT QSSPythonSources.Dialects.quir
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectQUIR.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
    MLIRQUIRDialect
    MLIRQUIRUtils
    MLIROQ3Dialect
    MLIRQCSDialect
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIQUIR
)

declare_mlir_python_extension(QSSPythonExtension.Dialects.QCS.Pybind
  MODULE_NAME _ibmDialectsQCS
  ADD_TO_PARENT QSSPythonSources.Dialects.qcs
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectQCS.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
    MLIRQCSDialect
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIQCS
)

declare_mlir_python_extension(QSSPythonExtension.Dialects.OQ3.Pybind
  MODULE_NAME _ibmDialectsOQ3
  ADD_TO_PARENT QSSPythonSources.Dialects.oq3
  ROOT_DIR "${PYTHON_SOURCE_DIR}"
  SOURCES
    DialectOQ3.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
    MLIROQ3Dialect
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    MLIRCAPIOQ3
)

add_mlir_python_common_capi_library(QSSPythonCAPI
  INSTALL_COMPONENT QSSPythonModules
  INSTALL_DESTINATION qss_compiler/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/qss_compiler/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../.."
  DECLARED_SOURCES
    QSSPythonSources
    MLIRPythonSources
    MLIRPythonExtension.AllPassesRegistration
)

add_mlir_python_modules(QSSPythonModules
  ROOT_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/qss_compiler/mlir/"
  INSTALL_PREFIX "qss_compiler/mlir/"
  DECLARED_SOURCES
    QSSPythonSources
    MLIRPythonSources
    MLIRPythonExtension.AllPassesRegistration
    MLIRPythonCAPIHeaderSources
  COMMON_CAPI_LINK_LIBS
    QSSPythonCAPI
)
