# (C) Copyright IBM 2021, 2023.
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

get_property(qssc_api_libs GLOBAL PROPERTY QSSC_API_LIBS)

set(llvm_code_gen_libraries
    LLVMAArch64CodeGen
    LLVMAArch64Info
    LLVMAArch64AsmParser
    LLVMAArch64Desc
    LLVMPowerPCCodeGen
    LLVMPowerPCInfo
    LLVMPowerPCAsmParser
    LLVMPowerPCDesc
    LLVMX86CodeGen
    LLVMX86Info
    LLVMX86AsmParser
    LLVMX86Desc
    LLVMX86TargetMCA
)

# Remove AArch64CodeGen if not present.
if (NOT LLVM_LIBRARIES MATCHES "LLVMAArch64CodeGen")
    list(REMOVE_ITEM llvm_code_gen_libraries
        LLVMAArch64CodeGen
        LLVMAArch64Info
        LLVMAArch64AsmParser
        LLVMAArch64Desc
    )
endif ()

qssc_add_plugin(QSSCTargetMock QSSC_TARGET_PLUGIN
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
QSSCHAL
MLIRExecutionEngine
MLIROptLib
MLIRLLVMIR
MLIRLLVMToLLVMIRTranslation
MLIRStandardOpsTransforms
${llvm_code_gen_libraries}
PLUGIN_REGISTRATION_HEADERS
${CMAKE_CURRENT_SOURCE_DIR}/Target.inc
)
