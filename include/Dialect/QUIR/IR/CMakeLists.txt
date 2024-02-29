# (C) Copyright IBM 2023, 2024.
#
# This code is part of Qiskit.
#
# This code is licensed under the Apache License, Version 2.0 with LLVM
# Exceptions. You may obtain a copy of this license in the LICENSE.txt
# file in the root directory of this source tree.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

add_mlir_dialect(QUIR quir)
add_mlir_doc(QUIR QUIRDialect generated/Dialect/QUIR/ -gen-dialect-doc -dialect=quir)

set(LLVM_TARGET_DEFINITIONS QUIR.td)
mlir_tablegen(QUIRAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=quir)
mlir_tablegen(QUIRAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=quir)
add_public_tablegen_target(QUIRAttributesIncGen)

set(LLVM_TARGET_DEFINITIONS QUIR.td)
mlir_tablegen(QUIREnums.h.inc -gen-enum-decls)
mlir_tablegen(QUIREnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(QUIREnumsIncGen)

# This dependency makes sure that the attributes and enums include files are
# generated before anything that requires the include files generated for the
# QUIR dialect
add_dependencies(MLIRQUIRIncGen QUIREnumsIncGen QUIRAttributesIncGen)

add_mlir_interface(QUIRInterfaces)
