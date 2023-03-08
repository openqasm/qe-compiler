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

ADD_LIBRARY(QSSCOpenQASM3Frontend OpenQASM3Frontend.cpp BaseQASM3Visitor.cpp PrintQASM3Visitor.cpp QUIRGenQASM3Visitor.cpp QUIRVariableBuilder.cpp)
include_directories(${OPENQASM_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)

target_link_libraries(QSSCOpenQASM3Frontend
        qasm::qasm
)

# enforce dependencies on QUIR tblgen generated headers
add_dependencies(QSSCOpenQASM3Frontend MLIRQUIRDialect)
