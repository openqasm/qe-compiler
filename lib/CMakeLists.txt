# (C) Copyright IBM 2021, 2023.
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

add_subdirectory(API)
add_subdirectory(Conversion)
add_subdirectory(Dialect)
add_subdirectory(Frontend)
add_subdirectory(HAL)
add_subdirectory(Payload)
add_subdirectory(Utils)

set(SOURCES
        QSSC.cpp
)
add_library(QSSCCore ${SOURCES})

get_property(qssc_targets GLOBAL PROPERTY QSSC_TARGETS)
get_property(mlir_dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(mlir_conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# All liibraries for the QSS-Compiler
set(qssc_api_libs
        ${qssc_targets}
        QSSCAPI
        QSSCHAL
        QSSCOpenQASM3Frontend
        QSSCPayload
        QSSCCore
        ${mlir_dialect_libs}
        ${mlir_conversion_libs}
        MLIRCallInterfaces
        MLIRExecutionEngine
        MLIROptLib
        MLIRQCSDialect
        MLIRQUIRDialect
        MLIRPulseDialect
        qasm::qasm
        mpc::mpc
        mpfr::mpfr
        gmp::gmp
        )
set_property(GLOBAL PROPERTY QSSC_API_LIBS ${qssc_api_libs})

# collect all required components in an "interface library", which exports its
# INTERFACE and dependencies to all targets that refer to the library yet does
# not materialize as an artifact by itself
add_library(QSSCLib INTERFACE)
target_link_libraries(QSSCLib INTERFACE ${qssc_api_libs})
