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

project(py_qssc)


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


macro(python_pkg_add_target_file file lib)
    set(src_file ${file})
    get_filename_component(barename ${file} NAME)
    set(dest_file ${CMAKE_CURRENT_BINARY_DIR}/qss_compiler/${barename})
    add_custom_command(OUTPUT ${dest_file}
      COMMAND ${CMAKE_COMMAND} -E copy
        ${src_file}
        ${dest_file}
      DEPENDS ${src_file}
    )
    LIST(APPEND ${lib} ${dest_file})
endmacro()


macro(python_pkg_add_target_files)
    foreach(file ${ARGN})
        python_pkg_add_target_file(${file} PY_LIB_FILES)
    endforeach()
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
)

foreach(dir ${QSSC_TARGET_PYTHON_DIRS})
    message(STATUS "Adding Python target directory: ${dir}")
    file(GLOB_RECURSE
            dir_files
            "${dir}/*py"
            )
    python_pkg_add_target_files("${dir_files}")
    unset(dir_files)
endforeach()

configure_file(pyproject.toml.in ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml)

add_subdirectory(qss_compiler)
add_custom_target(copy_python_files DEPENDS ${PY_LIB_FILES})
add_dependencies(py_qssc copy_python_files)
