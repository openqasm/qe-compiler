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

set (CXX_FLAGS
    -std=c++17
    -fno-strict-aliasing
    -fexceptions
    -fno-omit-frame-pointer
    -Werror
)

option(DETECT_TARGET_TRIPLE "Automatically detect the target triple for clang" ON)
if (DETECT_TARGET_TRIPLE)
execute_process (
    COMMAND bash -c "llvm-config --host-target | tr -d '\n'"
    OUTPUT_VARIABLE LLVM_TARGET_TRIPLE
)
list(APPEND CXX_FLAGS "-target ${LLVM_TARGET_TRIPLE}")
endif()

list (JOIN CXX_FLAGS " " CXX_FLAGS_STR)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_FLAGS_STR}")

# Google Sanitizers ------------------------------------------------------------
if(ENABLE_ADDRESS_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
endif()

if(ENABLE_UNDEFINED_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
endif()

if(ENABLE_THREAD_SANITIZER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
endif()

if(ENABLE_ADDRESS_SANITIZER OR ENABLE_UNDEFINED_SANITIZER OR ENABLE_THREAD_SANITIZER)
    # require flag to link shared libraries (e.g., py_qssc) with clang/LLVM and -fsanitize
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -shared-libsan")
endif()

set (CMAKE_CXX_FLAGS_DEBUG "-g3 -O0")
set (CMAKE_CXX_FLAGS_RELEASE "-g -O2 -DNOVERIFY")

set (CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
