# (C) Copyright IBM 2023.
#
# This code is part of Qiskit.
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

function(qssc_set_output_directory target)
    cmake_parse_arguments(ARG
        ""
        "BINARY_DIR;LIBRARY_DIR"
        ""
        ${ARGN}
    )
    if(ARG_BINARY_DIR)
        set_target_properties(${target} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${ARG_BINARY_DIR})
    endif()
    if(ARG_LIBRARY_DIR)
        set_target_properties(${target} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${ARG_LIBRARY_DIR})
    endif()
endfunction()


function(qssc_add_library name)
    cmake_parse_arguments(ARG
        "MODULE;SHARED;STATIC"
        "OUTPUT_NAME;"
        "ADDITIONAL_HEADERS;ADDITIONAL_HEADER_DIRS;DEPENDS;LINK_LIBS"
        ${ARGN})

    # Sources are unused arguments
    set(ALL_FILES ${ARG_UNPARSED_ARGUMENTS})

    list(APPEND QSSC_COMMON_DEPENDS ${ARG_DEPENDS})

    if(ARG_ADDITIONAL_HEADERS)
        set(ARG_ADDITIONAL_HEADERS ADDITIONAL_HEADERS ${ARG_ADDITIONAL_HEADERS})
    endif()

    find_all_header_files(FOUND_HEADERS "${ARG_ADDITIONAL_HEADER_DIRS}")
    list(APPEND ADDITIONAL_HEADERS ${ARG_ADDITIONAL_HEADERS} ${FOUND_HEADERS})
    set_source_files_properties(${ADDITIONAL_HEADERS} PROPERTIES HEADER_FILE_ONLY ON)
    list(APPEND ALL_FILES ${ADDITIONAL_HEADERS})

    message(STATUS "Sources for target: ${name} - ${ALL_FILES}")

    if((ARG_MODULE AND ARG_STATIC) OR ARG_OBJECT)
        set(obj_name "obj.${name}")
        add_library(${obj_name} OBJECT EXCLUDE_FROM_ALL
            ${ALL_FILES}
        )
        set(ALL_FILES "$<TARGET_OBJECTS:${obj_name}>" ${DUMMY_FILE})
        list(APPEND objlibs ${obj_name})
        set_target_properties(${obj_name} PROPERTIES FOLDER "Object Libraries")
        if(ARG_DEPENDS)
            add_dependencies(${obj_name} ${ARG_DEPENDS})
        endif()

        if(ARG_LINK_LIBS)
            foreach(link_lib ${ARG_LINK_LIBS})
                add_dependencies(${obj_name} ${link_lib})
            endforeach()
        endif()
    endif()

    if(ARG_SHARED AND ARG_STATIC)
        set(name_static "${name}_static")
        if(ARG_OUTPUT_NAME)
            set(output_name OUTPUT_NAME "${ARG_OUTPUT_NAME}")
        endif()

        qssc_add_library(${name_static} STATIC
            ${output_name}
            OBJLIBS ${ALL_FILES}
            LINK_LIBS ${ARG_LINK_LIBS}
        )
        set(ARG_STATIC)
    endif()

    if(ARG_MODULE)
        add_library(${name} MODULE ${ALL_FILES})
    elseif(ARG_SHARED)
        add_library(${name} SHARED ${ALL_FILES})
    else()
        add_library(${name} STATIC ${ALL_FILES})
    endif()

    set_output_directory(${name} BINARY_DIR ${QSSC_RUNTIME_OUTPUT_INTDIR} LIBRARY_DIR ${QSSC_LIBRARY_OUTPUT_INTDIR})

    if(ARG_OUTPUT_NAME)
        set_target_properties(${name}
            PROPERTIES
            OUTPUT_NAME ${ARG_OUTPUT_NAME}
        )
    endif()

    if(ARG_MODULE)
        set_target_properties(${name} PROPERTIES
            PREFIX ""
            SUFFIX ${QSSC_PLUGIN_EXT}
            )
    endif()

    if(ARG_STATIC)
        set(libtype PUBLIC)
    else()
        set(libtype PRIVATE)
    endif()

    target_link_libraries(${name}
        ${libtype}
        ${ARG_LINK_LIBS}
    )

endfunction()

# Find all headers in all directories.
function(find_all_header_files headers_out header_dirs)
    glob_headers(headers *.h)
    list(APPEND all_headers ${headers})

    foreach(dir ${header_dirs})
        file(GLOB headers "${dir}/*.h")
        list(APPEND all_headers ${headers})
        file(GLOB headers "${dir}/*.inc")
        list(APPEND all_headers ${headers})
    endforeach()

    set(${headers_out} ${all_headers} PARENT_SCOPE)
endfunction()

# NOTE: It is not ideal to have a separate property defined for
# each plugin (the number could grow large). However, it is in
# our better interest--at least, for now--to keep distinct plugin
# types separate to prevent unecessary linking.
define_property(GLOBAL PROPERTY QSSC_TARGETS
    BRIEF_DOCS "QSSC system targets to be built"
    FULL_DOCS "QSSC system targets to be built")
# Initialize property
set_property(GLOBAL PROPERTY QSSC_TARGETS "")

define_property(GLOBAL PROPERTY QSSC_TARGET_REGISTRATION_HEADERS
        BRIEF_DOCS "QSSC system target registration headers"
        FULL_DOCS "QSSC system target registration headers")
# Initialize property
set_property(GLOBAL PROPERTY QSSC_TARGET_REGISTRATION_HEADERS "")

define_property(GLOBAL PROPERTY QSSC_PAYLOADS
    BRIEF_DOCS "QSSC payload plugins to be built"
    FULL_DOCS "QSSC payload plugins to be built")
# Initialize property
set_property(GLOBAL PROPERTY QSSC_PAYLOADS "")

define_property(GLOBAL PROPERTY QSSC_PAYLOAD_REGISTRATION_HEADERS
        BRIEF_DOCS "QSSC payload plugin registration headers"
        FULL_DOCS "QSSC payload plugin registration headers")
# Initialize property
set_property(GLOBAL PROPERTY QSSC_PAYLOAD_REGISTRATION_HEADERS "")

# Add a LIT test suite using the QSS LIT config.
function(qssc_add_lit_test_suite target_name suite_name tests_directory extra_args)
    set(QSSC_LIT_SUITE_NAME "${suite_name}")
    set(QSSC_LIT_SOURCE_DIR "${tests_directory}")
    set(QSSC_LIT_EXEC_DIR "${CMAKE_CURRENT_BINARY_DIR}/${target_name}")
    configure_lit_site_cfg(
            "${QSSC_TEST_DIR}/lit.site.cfg.py.in"
            "${QSSC_LIT_EXEC_DIR}/lit.site.cfg.py"
            MAIN_CONFIG
            "${QSSC_TEST_DIR}/lit.cfg.py"
    )

    set(QSS_COMPILER_TEST_DEPENDS
            FileCheck count not
            qss-compiler
            qss-opt
            )

    add_lit_testsuite(${target_name} "Running the ${suite_name} regression tests"
            "${CMAKE_CURRENT_BINARY_DIR}/${target_name}/"
            DEPENDS ${QSS_COMPILER_TEST_DEPENDS}
            ARGS ${extra_args}
            )

    set_target_properties(${target_name} PROPERTIES FOLDER "Tests")
    add_dependencies(check-qss-compiler "${target_name}")

endfunction(qssc_add_lit_test_suite)

# Add a QSS Compiler plugin
#
# qssc_add_plugin(plugin_name plugin_type
#    <List of sources>
#
#    [ADDITIONAL_HEADERS
#     <List of additional header files ...> ]
#
#    [ADDITIONAL_HEADER_DIRS]
#     <List of additional include directories ...> ]
#
#    [LINK_LIBS
#     <List of libraries to link ...> ]
#
#    [CUSTOM_RESOURCES
#     <List of cmake targets that provide static resources for the target
#     system...>
#
#     Each static resource must be defined as cmake target.
#
#     By default, the plugin's name is used as the resource's file name. To
#     override that, define a plugin property RESOURCE_OUTPUT_NAME that defines
#     the file name of the resource in the build directory and when installed.
#
#     If the resource is an executable, the plugin property RESOURCE_IS_PROGRAM
#     must be set to true.
#
#    [PLUGIN_REGISTRATION_HEADERS
#     <List of headers to be included by the QSS compiler plugin registry
#     ...>
#
#     Used to register plugins via static initialization.
#
#    [PLUGIN_SHORT_NAME
#     <optional short name for the plugin that will be used
#      in path names for resources -- must match <Plugin>::getName()
#      so that resources can be found at runtime ]
# )
function(qssc_add_plugin plugin_name plugin_type)
  # Ensure a supported plugin type was specified
  set(SUPPORTED_PLUGIN_TYPES QSSC_PAYLOAD_PLUGIN QSSC_TARGET_PLUGIN)
  string(TOUPPER ${plugin_type} PLUGIN_TYPE_TOUPPER)
  if(NOT ${PLUGIN_TYPE_TOUPPER} IN_LIST SUPPORTED_PLUGIN_TYPES)
    string(REPLACE ";" ", " supported "${SUPPORTED_PLUGIN_TYPES}")
    message(FATAL_ERROR
      "${plugin_type} is an unsupported plugin type.\n"
      "Supported plugin types are: ${supported}"
    )
  else()
    # Parse `PLUGIN_TYPE_TOUPPER` to get just type name (e.g., 'payload',
    # 'target') to use for setting variable names (TOUPPER) and paths (TOLOWER)
    string(REPLACE "_" ";"
      PLUGIN_TYPE_TOUPPER_AS_LIST "${PLUGIN_TYPE_TOUPPER}"
    )
    list(GET PLUGIN_TYPE_TOUPPER_AS_LIST 1 PLUGIN_TYPE_TOUPPER)
    string(TOLOWER ${PLUGIN_TYPE_TOUPPER} PLUGIN_TYPE_TOLOWER)
  endif()
  include_directories(BEFORE
    ${CMAKE_CURRENT_SOURCE_DIR})

  # peel off arguments specific to the plugin
  cmake_parse_arguments(ARG
    ""
    "PLUGIN_SHORT_NAME;"
    "PLUGIN_REGISTRATION_HEADERS;CUSTOM_RESOURCES"
    ${ARGN}
  )

  qssc_add_library(${plugin_name} ${ARG_UNPARSED_ARGUMENTS})

  message(STATUS "Adding ${PLUGIN_TYPE_TOLOWER} plugin: ${plugin_name}")

  if(PLUGIN_TYPE_TOLOWER STREQUAL "target")
    # enforce dependency on tblgen generated headers
    add_dependencies(${plugin_name} MLIRQUIRDialect)
  endif()

  # add registration header(s) to global var
  foreach(REG_HEADER ${ARG_PLUGIN_REGISTRATION_HEADERS})
      message(STATUS "Checking for registration header: ${REG_HEADER}")
      if(NOT EXISTS "${REG_HEADER}")
          message(FATAL_ERROR "Missing registration header for plugin ${plugin_name}: ${REG_HEADER}")
      else()
          set_property(GLOBAL APPEND PROPERTY QSSC_${PLUGIN_TYPE_TOUPPER}_REGISTRATION_HEADERS "${REG_HEADER}")
      endif()
  endforeach()

  set_property(GLOBAL APPEND PROPERTY QSSC_${PLUGIN_TYPE_TOUPPER}S "${plugin_name}")

  if(ARG_PLUGIN_SHORT_NAME)
      message(STATUS "Using short name ${ARG_PLUGIN_SHORT_NAME} for ${plugin_name} in resource paths")
      set(plugin_name_resources ${ARG_PLUGIN_SHORT_NAME})
  else()
      set(plugin_name_resources ${plugin_name})
  endif()

  if(ARG_CUSTOM_RESOURCES)
      add_dependencies(${plugin_name} ${ARG_CUSTOM_RESOURCES})

      # define build directory for the plugin's resources
      # - attach as  property RESOURCE_OUTPUT_DIRECTORY to each resource target
      set(QSSC_${PLUGIN_TYPE_TOUPPER}_${plugin_name}_RESOURCE_DIR ${QSSC_RESOURCES_OUTPUT_INTDIR}/${PLUGIN_TYPE_TOLOWER}s/${plugin_name_resources})
      file(MAKE_DIRECTORY ${QSSC_${PLUGIN_TYPE_TOUPPER}_${plugin_name}_RESOURCE_DIR})
      set_target_properties(${ARG_CUSTOM_RESOURCES} PROPERTIES RESOURCE_OUTPUT_DIRECTORY ${QSSC_${PLUGIN_TYPE_TOUPPER}_${plugin_name}_RESOURCE_DIR})

      foreach(resource ${ARG_CUSTOM_RESOURCES})
          get_target_property(RESOURCE_OUTPUT_NAME ${resource} RESOURCE_OUTPUT_NAME)
          if (NOT RESOURCE_OUTPUT_NAME)
              set(RESOURCE_OUTPUT_NAME ${resource})
          endif()
          get_target_property(RESOURCE_IS_PROGRAM ${resource} RESOURCE_IS_PROGRAM)
          set(resource_file ${QSSC_${PLUGIN_TYPE_TOUPPER}_${plugin_name}_RESOURCE_DIR}/${RESOURCE_OUTPUT_NAME})
          set(resource_destination ${QSSC_RESOURCES_INSTALL_PREFIX}/${PLUGIN_TYPE_TOLOWER}s/${plugin_name_resources})

          if(RESOURCE_IS_PROGRAM)
              install(PROGRAMS ${resource_file}
                      DESTINATION ${resource_destination}
              )
          else()
              install(FILES ${resource_file}
                      DESTINATION ${resource_destination}
              )
          endif()
      endforeach()
  endif()
endfunction(qssc_add_plugin plugin_name plugin_type)

include(GoogleTest)
# From: https://cliutils.gitlab.io/modern-cmake/chapters/testing/googletest.html
macro(package_add_test TESTNAME)
    # create an exectuable in which the tests will be stored
    add_executable(${TESTNAME} ${ARGN})
    # link the Google test infrastructure, mocking library, and a default main fuction to
    # the test executable.  Remove g_test_main if writing your own main function.
    target_link_libraries(${TESTNAME} GTest::gtest_main GTest::gtest)
    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TESTNAME}
        # set a working directory so your project root so that you can find test data via paths relative to the project root
        WORKING_DIRECTORY ${PROJECT_DIR}
        PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_DIR}"
        DISCOVERY_TIMEOUT 300 # Prevent timeout errors on test discovery during build
    )
    set_target_properties(${TESTNAME} PROPERTIES FOLDER tests)
    list(APPEND QSSC_UNITTESTS ${TESTNAME})
endmacro()

# add a google test with libraries
macro(package_add_test_with_libs TESTNAME)
    cmake_parse_arguments(ARG
            ""
            ""
            "LIBRARIES"
            ${ARGN}
    )
    package_add_test(${TESTNAME} ${ARG_UNPARSED_ARGUMENTS})
    target_link_libraries(${TESTNAME} GTest::gtest_main GTest::gtest ${ARG_LIBRARIES})
endmacro()


# Version adapter for add_mlir_doc during transition from LLVM 12 to newer
# (order of parameters differs)
function(qssc_add_mlir_doc doc_filename output_file output_directory command)
    if("${LLVM_VERSION_MAJOR}" EQUAL "12")
        add_mlir_doc(${doc_filename} ${command} ${output_file} ${output_directory})
    else()
        add_mlir_doc(${doc_filename} ${output_file} ${output_directory} ${command})
    endif()
endfunction()
