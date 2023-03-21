# (C) Copyright IBM 2022, 2023.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

add_mlir_dialect_library(MLIRQUIRUtils

    Utils.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/QUIR

    DEPENDS
    MLIRQUIRIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
