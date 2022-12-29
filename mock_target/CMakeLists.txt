# (C) Copyright IBM 2021, 2022.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

get_property(qssc_api_libs GLOBAL PROPERTY QSSC_API_LIBS)

qssc_add_system_target(QSSCTargetMock
Conversion/QUIRToStandard/QUIRToStandard.cpp
MockTarget.cpp
MockUtils.cpp
Transforms/FunctionLocalization.cpp
Transforms/QubitLocalization.cpp

ADDITIONAL_HEADER_DIRS
${CMAKE_CURRENT_SOURCE_DIR}
${QSSC_INCLUDE_DIR}/HAL

LINK_LIBS
${qssc_api_libs}
MLIRExecutionEngine
MLIROptLib
MLIRLLVMIR
MLIRLLVMToLLVMIRTranslation
MLIRStandardOpsTransforms
LLVMPowerPCCodeGen
LLVMX86CodeGen

TARGET_REGISTRATION_HEADERS
${CMAKE_CURRENT_SOURCE_DIR}/Target.inc
)
