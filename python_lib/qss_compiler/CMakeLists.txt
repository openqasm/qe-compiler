set(SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/lib.cpp
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
                link.py
                __init__.py
)

set_target_properties(
        py_qssc
        PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/
        LIBRARY_OUTPUT_DIRECTORY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/
        )

# require a target for copying files in each directory
add_custom_target(copy_python_files_qss_compiler DEPENDS ${PY_QSSC_FILES})
add_dependencies(py_qssc copy_python_files_qss_compiler)

# Make static resources of the compiler available in the python package
file(CREATE_LINK ${QSSC_RESOURCES_OUTPUT_INTDIR} ${CMAKE_CURRENT_BINARY_DIR}/resources SYMBOLIC)
