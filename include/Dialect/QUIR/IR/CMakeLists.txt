# (C) Copyright IBM 2022.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

add_mlir_dialect(QUIR quir)
qssc_add_mlir_doc(QUIR QUIRDialect generated/Dialect/QUIR/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS QUIR.td)
mlir_tablegen(QUIRAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=quir)
mlir_tablegen(QUIRAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=quir)

set(LLVM_TARGET_DEFINITIONS QUIRPatterns.td)
mlir_tablegen(QUIRPatterns.inc -gen-rewriters)
add_public_tablegen_target(MLIRQUIRPatternsIncGen)
