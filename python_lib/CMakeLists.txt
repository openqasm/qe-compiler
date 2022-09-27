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
