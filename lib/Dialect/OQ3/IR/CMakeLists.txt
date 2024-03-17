# (C) Copyright IBM 2023.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

add_mlir_dialect_library(MLIROQ3Dialect
        OQ3Dialect.cpp
        OQ3Ops.cpp
        OQ3Patterns.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/OQ3

        DEPENDS
        MLIROQ3PatternsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRMathDialect
	)
