# (C) Copyright IBM 2021, 2023.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

#get_property(qssc_api_libs GLOBAL PROPERTY QSSC_API_LIBS)
#
#set(llvm_code_gen_libraries
#        LLVMAArch64CodeGen
#        LLVMAArch64Info
#        LLVMAArch64AsmParser
#        LLVMAArch64Desc
#        LLVMPowerPCCodeGen
#        LLVMPowerPCInfo
#        LLVMPowerPCAsmParser
#        LLVMPowerPCDesc
#        LLVMX86CodeGen
#        LLVMX86Info
#        LLVMX86AsmParser
#        LLVMX86Desc
#        LLVMX86TargetMCA
#        )
#
## Remove AArch64CodeGen if not present.
#if (NOT LLVM_LIBRARIES MATCHES "LLVMAArch64CodeGen")
#    list(REMOVE_ITEM llvm_code_gen_libraries
#            LLVMAArch64CodeGen
#            LLVMAArch64Info
#            LLVMAArch64AsmParser
#            LLVMAArch64Desc
#            )
#endif ()

set(QSSC_PAYLOAD_DIRS
        ${QSSC_PAYLOAD_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        )

qssc_add_plugin(QSSCPayloadZip QSSC_PAYLOAD_PLUGIN
        ZipPayload.cpp

        ADDITIONAL_HEADER_DIRS
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${QSSC_INCLUDE_DIR}/Payload

        LINK_LIBS
        QSSCPayload
        libzip::libzip

        PLUGIN_REGISTRATION_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/Payload.inc
        )
