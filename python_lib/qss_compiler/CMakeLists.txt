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

set(SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/lib.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/lib_enums.cpp
)

pybind11_add_module(py_qssc SHARED ${SOURCES})

target_link_libraries(py_qssc PRIVATE QSSCLib)

# collect python package files from this directory
# into a variable PY_QSSC_FILES
macro(python_pkg_add_files)
    foreach(file ${ARGN})
        python_pkg_add_file(${file} PY_QSSC_FILES)
    endforeach()
endmacro()

# Copy over python source files into build directory so that the python package
# ("wheel") can be built there.
# Note that when adding new source files, you need to add them here.
python_pkg_add_files(
                compile.py
                exceptions.py
                link.py
                __init__.py
)

set_target_properties(
        py_qssc
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/
        LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_CURRENT_BINARY_DIR}/
        LIBRARY_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_CURRENT_BINARY_DIR}/
        )

# require a target for copying files in each directory
add_custom_target(copy_python_files_qss_compiler DEPENDS ${PY_QSSC_FILES})
add_dependencies(py_qssc copy_python_files_qss_compiler)

# Make static resources of the compiler available in the python package
file(CREATE_LINK ${QSSC_RESOURCES_OUTPUT_INTDIR} ${CMAKE_CURRENT_BINARY_DIR}/resources SYMBOLIC)
