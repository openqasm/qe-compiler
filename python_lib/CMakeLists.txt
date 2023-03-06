# (C) Copyright IBM 2023.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

configure_file(pyproject.toml.in ${CMAKE_CURRENT_BINARY_DIR}/pyproject.toml)

add_subdirectory(qss_compiler)
add_custom_target(copy_python_files DEPENDS ${PY_LIB_FILES})
add_dependencies(py_qssc copy_python_files)
