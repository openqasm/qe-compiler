# (C) Copyright IBM 2023.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

add_mlir_conversion_library(OQ3ToStandard
OQ3ToStandard.cpp

ADDITIONAL_HEADER_DIRS
${PROJECT_SOURCE_DIR}/include/Conversion/OQ3ToStandard/

LINK_COMPONENTS
Core

LINK_LIBS PUBLIC
MLIRIR
MLIROQ3Dialect
)
