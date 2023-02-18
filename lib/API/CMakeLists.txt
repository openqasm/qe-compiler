# (C) Copyright IBM 2021, 2023.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


add_library(QSSCAPI api.cpp error.cpp)

set(LIBS
        qasm::qasm
        )
target_link_libraries(QSSCAPI ${LIBS})

add_dependencies(QSSCAPI MLIROQ3Dialect MLIRQCSDialect MLIRQUIRDialect)
