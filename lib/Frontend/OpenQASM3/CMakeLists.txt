# (C) Copyright IBM 2021.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

ADD_LIBRARY(QSSCVisitor BaseQASM3Visitor.cpp PrintQASM3Visitor.cpp QUIRGenQASM3Visitor.cpp QUIRVariableBuilder.cpp)
include_directories(${OPENQASM_INCLUDE_DIR})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${gmp_INCLUDES})

target_link_libraries(QSSCVisitor
        qasm::qasm
)

# enforce dependencies on QUIR tblgen generated headers
add_dependencies(QSSCVisitor MLIRQUIRDialect)
