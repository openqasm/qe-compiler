# (C) Copyright IBM 2023.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

add_mlir_dialect(QuSysOps qusys)
qssc_add_mlir_doc(QuSysOps QuSysOps generated/Dialect/QuSys/ -gen-dialect-doc)

set(LLVM_TARGET_DEFINITIONS QuSysDialect.td)
