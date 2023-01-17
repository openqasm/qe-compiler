# (C) Copyright IBM 2023.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

add_mlir_dialect(System sys)
qssc_add_mlir_doc(System System generated/Dialect/System/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS System.td)
mlir_tablegen(SystemAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=sys)
mlir_tablegen(SystemAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=sys)

add_mlir_interface(SystemInterfaces)
