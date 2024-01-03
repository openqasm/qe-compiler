# (C) Copyright IBM 2023.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

add_mlir_dialect(OQ3Ops oq3)
add_mlir_doc(OQ3Ops OQ3Ops generated/Dialect/OQ3/ -gen-dialect-doc -dialect=oq3)

set(LLVM_TARGET_DEFINITIONS OQ3Dialect.td)

set(LLVM_TARGET_DEFINITIONS OQ3Patterns.td)
mlir_tablegen(OQ3Patterns.inc -gen-rewriters)
add_public_tablegen_target(MLIROQ3PatternsIncGen)
