# (C) Copyright IBM 2021.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

cmake_minimum_required(VERSION 3.13.4...3.17)

# ignore the RPATH settings for install_name on macOS.
# https://cmake.org/cmake/help/latest/policy/CMP0068.html
if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

# honor CMAKE_REQUIRED_LIBRARIES in the include file check macros
# https://cmake.org/cmake/help/latest/policy/CMP0075.html
if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

# do nothing when a normal variable name collides with that of a
# cached variable.
# https://cmake.org/cmake/help/latest/policy/CMP0077.html
if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

# Ninja generators transform DEPFILE s from add_custom_command().
# applies to MLIR's calls to TableGen. LLVM upstream uses OLD behavior for
# CMP0116.
# see https://cmake.org/cmake/help/latest/policy/CMP0116.html
# see https://reviews.llvm.org/D101083
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()


enable_language(C CXX ASM)

# Find external packages
find_package(Python COMPONENTS Interpreter Development)
find_package(qasm REQUIRED)
find_package(mpc REQUIRED)
find_package(mpfr REQUIRED)
find_package(gmp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(libzip REQUIRED)
find_package(GTest REQUIRED)
find_package(LLVM REQUIRED CONFIG)

find_package(clang-tools-extra REQUIRED CONFIG)

if(DEFINED ENV{VERSION_STRING})
    set(VERSION_STRING $ENV{VERSION_STRING})
else()
    # Get the SCM version from whichever repo contains the root CMakeLists.txt.
    execute_process(
        COMMAND git rev-parse --show-toplevel
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE QSS_ROOT_GIT_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Root CMake project Git directory (used for version): ${QSS_ROOT_GIT_DIR}")
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "from setuptools_scm import get_version; print(get_version(root='${QSS_ROOT_GIT_DIR}'))"
        OUTPUT_VARIABLE VERSION_STRING
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    unset(QSS_ROOT_GIT_DIR)
endif()

message(STATUS "SCM Version: ${VERSION_STRING}")

# Version information
string(REGEX MATCHALL "^([0-9]+)\\.([0-9]+)\\.([a-zA-Z0-9\+_.-]+)" _ ${VERSION_STRING})
set(QSSC_VERSION_MAJOR ${CMAKE_MATCH_1})
message(STATUS "QSS Compiler Major Version: ${QSSC_VERSION_MAJOR}")
set(QSSC_VERSION_MINOR ${CMAKE_MATCH_2})
message(STATUS "QSS Compiler Minor Version: ${QSSC_VERSION_MINOR}")
set(QSSC_VERSION_PATCH ${CMAKE_MATCH_3})
message(STATUS "QSS Compiler Patch Version: ${QSSC_VERSION_PATCH}")
set(QSSC_VERSION ${QSSC_VERSION_MAJOR}.${QSSC_VERSION_MINOR}.${QSSC_VERSION_PATCH})
message(STATUS "QSS Compiler Version: ${QSSC_VERSION}")

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler options
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/apple-clang.cmake)
    else()
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/llvm-clang.cmake)
    endif()
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        message(FATAL_ERROR "Building with GCC on MacOS is not yet supported.")
    else()
        include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/linux-gcc.cmake)
    endif()
endif()

project(qss-compiler)
project(qss-opt LANGUAGES CXX C)

# ------------------------------------------------------------------------------
# Project configuration
# ------------------------------------------------------------------------------
option(CI_RUN "This is a CI run" OFF)
option(PACKAGE_TESTS "Build the tests" ON)
option(ENABLE_ADDRESS_SANITIZER "Compile with google address sanitizer" ${CI_RUN})
option(ENABLE_UNDEFINED_SANITIZER "Compile with google undefined sanitizer" ${CI_RUN})
option(ENABLE_THREAD_SANITIZER "Compile with google thread sanitizer" ${CI_RUN})
option(TEST_USE_VG "Run qss-compiler tests under Valgrind" ${CI_RUN})

set(QSSC_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "QSSC_SRC_DIR")
set(QSSC_INCLUDE_DIR ${QSSC_SRC_DIR}/include CACHE INTERNAL "QSSC_INCLUDE_DIR")
set(QSSC_TEST_DIR ${QSSC_SRC_DIR}/test CACHE INTERNAL "QSSC_TEST_DIR")
set(QSSC_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR} CACHE INTERNAL "QSSC_BINARY_DIR")
set(QSSC_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin)
set(QSSC_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib)
set(QSSC_RESOURCES_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/resources)
set(QSSC_PLUGIN_EXT ${CMAKE_SHARED_LIBRARY_SUFFIX})

option(QSSC_WITH_MOCK_TARGET "Available targets include the built-in mock target" ON)
set(QSSC_TARGET_DIRS
        ""
        CACHE STRING
        "List of directories containing QSS compiler targets."
)

set(QSSC_TARGET_TEST_DIRS
        ""
        CACHE STRING
        "List of directories containing QSS compiler LIT test suites."
        )

if(QSSC_WITH_MOCK_TARGET)
    set(QSSC_TARGET_DIRS
            ${QSSC_TARGET_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}/mock_target
    )
    set(QSSC_TARGET_TEST_DIRS
            ${QSSC_TARGET_TEST_DIRS}
            ${CMAKE_CURRENT_SOURCE_DIR}/mock_target/test
    )
endif()

set(QSSC_RESOURCES_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}/lib/qssc)
configure_file(include/Config.h.in ${QSSC_BINARY_DIR}/include/Config.h)

# prepare variables for build environment
configure_file(qssc-activate.in ${QSSC_BINARY_DIR}/qssc-activate @ONLY)

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
file(GLOB_RECURSE
  ALL_CXX_INCLUDE_FILES
  include/*cpp include/*h
)
file(GLOB_RECURSE
  ALL_CXX_LIB_FILES
  lib/*cpp lib/*h
)
file(GLOB_RECURSE
  ALL_CXX_QSSC_FILES
  tools/qss-compiler/*cpp tools/qss-compiler/*h
  tools/qss-opt/*cpp tools/qss-opt/*h
)
file(GLOB_RECURSE
  ALL_CXX_TEST_FILES
  test/*cpp test/*h
)

foreach(dir ${QSSC_TARGET_DIRS})
    file(GLOB_RECURSE
            dir_files
            "${dir}/*cpp" "${dir}/*h"
    )
    list(APPEND ALL_CXX_TARGET_FILES "${dir_files}")
    unset(dir_files)
endforeach()

foreach(dir ${QSSC_TARGET_TEST_DIRS})
    file(GLOB_RECURSE
            dir_files
            "${dir}/*cpp" "${dir}/*h"
            )
    list(APPEND ALL_CXX_TARGET_TEST_FILES "${dir_files}")
    unset(dir_files)
endforeach()

# All CXX source files
set(QSSC_ALL_CXX_SOURCE_FILES
    ${ALL_CXX_INCLUDE_FILES}
    ${ALL_CXX_LIB_FILES}
    ${ALL_CXX_QSSC_FILES}
    ${ALL_CXX_TEST_FILES}
    ${ALL_CXX_TARGET_FILES}
    ${ALL_CXX_TARGET_TEST_FILES}
    CACHE INTERNAL "QSSC_ALL_CXX_SOURCE_FILES"
)

# Clang Format -----------------------------------------------------------------

find_program(CLANG_FORMAT_BIN clang-format)

add_custom_target(
    fix-format
    COMMAND ${CLANG_FORMAT_BIN}
    --style=file
    -i ${QSSC_ALL_CXX_SOURCE_FILES}
    COMMENT "running clang-format"
)

add_custom_target(
    check-format
    COMMAND ${PROJECT_SOURCE_DIR}/check_format.sh ${CLANG_FORMAT_BIN} ${QSSC_ALL_CXX_SOURCE_FILES}
    COMMENT "running clang-format check"
)

# Clang Tidy - Linting ----------------------------------------------------------
find_program(CLANG_TIDY_BIN clang-tidy)
find_program(CLANG_AR_BIN clang-apply-replacements REQUIRED)
find_program(RUN_CLANG_TIDY_BIN run-clang-tidy)

list(APPEND RUN_CLANG_TIDY_BIN_ARGS
    -clang-tidy-binary ${CLANG_TIDY_BIN}
    -clang-apply-replacements-binary ${CLANG_AR_BIN}
    -style=file
    -quiet
    -header-filter="${CMAKE_SOURCE_DIR}/[^b].*"
)

# run clang-tidy linter only when compiling with clang (compiler flags may be
# incompatible from gcc builds)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
add_custom_target(
    check-tidy
    COMMAND ${RUN_CLANG_TIDY_BIN} ${RUN_CLANG_TIDY_BIN_ARGS}
    COMMENT "running linter (clang tidy)"
)
else()
add_custom_target(
    check-tidy
    COMMENT "skip linter (clang tidy) in non-clang builds"
)
endif()

add_custom_target(
    fix-tidy
    COMMAND ${RUN_CLANG_TIDY_BIN} ${RUN_CLANG_TIDY_BIN_ARGS} -fix
    COMMENT "running linter (clang tidy) and applying fixes"
    )

# ------------------------------------------------------------------------------
# Analysis
# ------------------------------------------------------------------------------

# Valgrind ----------------------------------------------------------------------
option(TEST_USE_VG "Run qss-compiler tests under Valgrind" ${CI_RUN})
if(TEST_USE_VG)
  # Use with standard test suite
  set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --leak-check=full")
  set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --track-fds=yes")
  set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --trace-children=yes")
  set(MEMORYCHECK_COMMAND_OPTIONS "${MEMORYCHECK_COMMAND_OPTIONS} --error-exitcode=1")
  # Use with lit test suite
  set(LIT_TEST_EXTRA_ARGS "${LIT_TEST_EXTRA_ARGS} --vg")
endif ()

# ------------------------------------------------------------------------------
# third-party
# ------------------------------------------------------------------------------
include_directories(${QSSC_INCLUDE_DIR})
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/AddQSSC.cmake)

# QASM -------------------------------------------------------------------------
set(OPENQASM_CMAKE_PACKAGE ON)
set(OPENQASM_WITH_TESTS OFF)
set(OPENQASM_WERROR OFF CACHE BOOL "Turn off Werror as not passing.")
set(BUILD_STATIC_LIBS ON CACHE BOOL "Turn on building static libs.")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Turn off building shared libs.")

# MLIR/LLVM -------------------------------------------------------------------
find_path(LLVM_MODULES NAMES TableGen.cmake AddLLVM.cmake HandleLLVMOptions.cmake PATH_SUFFIXES llvm REQUIRED)
list(PREPEND CMAKE_MODULE_PATH "${LLVM_MODULES}")
message(STATUS "Using LLVM modules from: ${LLVM_MODULES}")

find_path(MLIR_MODULES NAMES AddMLIR.cmake PATH_SUFFIXES mlir REQUIRED)
list(PREPEND CMAKE_MODULE_PATH "${MLIR_MODULES}")
message(STATUS "Using MLIR modules from: ${MLIR_MODULES}")

find_package(MLIR REQUIRED)
find_package(LLD REQUIRED)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using LLDConfig.cmake in: ${LLD_DIR}")
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})


include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include(LLVMExports)

include_directories(BEFORE SYSTEM ${LLVM_INCLUDE_DIRS})
include_directories(BEFORE SYSTEM ${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)


set(LLVM_ENABLE_RTTI ON)
set(LLVM_ENABLE_EH ON)


# Pybind11 -------------------------------------------------------------------
find_package(pybind11 REQUIRED)

# nlohmann_json -------------------------------------------------------------------
include_directories(BEFORE SYSTEM ${nlohmann_json_INCLUDE_DIRS})

# Boost -------------------------------------------------------------------
find_package(Boost REQUIRED)
include_directories(BEFORE SYSTEM ${Boost_INCLUDE_DIRS})

# ------------------------------------------------------------------------------
# QSS-COMPILER
# ------------------------------------------------------------------------------
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(python_lib)

# Analysis ----------------------------------------------------------------------
add_custom_target(check-analysis DEPENDS check-tidy check-format COMMENT "Running all analysis")

# Intallation -------------------------------------------------------------------
install(TARGETS qss-compiler qss-opt DESTINATION bin)

# Tests -------------------------------------------------------------------------
if(PACKAGE_TESTS)
    include(CTest)
    add_subdirectory(test)
endif()

include(CTest)
enable_testing()

# Source files, without extensions or directory names
# Here assuming each file has both header and impl
set(SRC_FILES QSSC)

# Create SOURCE_FILES
foreach(file ${SRC_FILES})
	list(APPEND SOURCE_FILES
        ${QSSC_SRC_DIR}/lib/${file}.cpp
        ${QSSC_INCLUDE_DIR}/${file}.h
	)
endforeach()

# Set up such that your IDE organizes the files correctly
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${SOURCE_FILES})

add_library(qss-compiler-opt ${SOURCE_FILES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# Check all --------------------------------------------------------------------
add_custom_target(check-all DEPENDS check-tests check-analysis build-docs COMMENT "Running all checks")
