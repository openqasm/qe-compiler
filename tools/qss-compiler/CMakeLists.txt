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
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.


add_llvm_executable(qss-compiler qss-compiler.cpp)
llvm_update_compile_flags(qss-compiler)
target_link_libraries(qss-compiler PRIVATE QSSCLib)
mlir_check_all_link_libraries(qss-compiler)
