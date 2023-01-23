# (C) Copyright IBM 2023.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

add_mlir_dialect(QCSOps qcs)
qssc_add_mlir_doc(QCSOps QCSOps generated/Dialect/QCS/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS QCSDialect.td)
