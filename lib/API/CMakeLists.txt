# (C) Copyright IBM 2021.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


add_library(QSSCAPI api.cpp)

set(LIBS
        qasm::qasm
        )
target_link_libraries(QSSCAPI ${LIBS})
target_include_directories(QSSCAPI PUBLIC ${gmp_INCLUDES})

add_dependencies(QSSCAPI MLIRQUIRDialect)
