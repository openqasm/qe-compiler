# (C) Copyright IBM 2022, 2023.
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

add_mlir_dialect(QUIR quir)
qssc_add_mlir_doc(QUIR QUIRDialect generated/Dialect/QUIR/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS QUIR.td)
mlir_tablegen(QUIRAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=quir)
mlir_tablegen(QUIRAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=quir)

set(LLVM_TARGET_DEFINITIONS QUIRPatterns.td)
mlir_tablegen(QUIRPatterns.inc -gen-rewriters)
add_public_tablegen_target(MLIRQUIRPatternsIncGen)

add_mlir_interface(QUIRInterfaces)
