# (C) Copyright IBM 2022.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

add_mlir_conversion_library(QUIRToStandard
CBitOperations.cpp
QUIRCast.cpp
SwitchOpLowering.cpp
TypeConversion.cpp
VariablesToGlobalMemRefConversion.cpp

ADDITIONAL_HEADER_DIRS
${PROJECT_SOURCE_DIR}/include/Conversion/QUIRToStandard/

LINK_COMPONENTS
Core

LINK_LIBS PUBLIC
MLIRIR
MLIRQUIRDialect
)
