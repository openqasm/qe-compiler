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


ADD_LIBRARY(QSSCOpenQASM3Frontend OpenQASM3Frontend.cpp BaseQASM3Visitor.cpp PrintQASM3Visitor.cpp QUIRGenQASM3Visitor.cpp QUIRVariableBuilder.cpp)
include_directories(${OPENQASM_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)

target_link_libraries(QSSCOpenQASM3Frontend
        qasm::qasm
)

# enforce dependencies on QUIR tblgen generated headers
add_dependencies(QSSCOpenQASM3Frontend MLIRQUIRDialect)
