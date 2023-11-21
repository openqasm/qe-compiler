# (C) Copyright IBM 2023.
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

add_mlir_dialect(Pulse pulse)
add_mlir_doc(Pulse PulseDialect generated/Dialect/Pulse/ -gen-dialect-doc -dialect=pulse)

add_mlir_interface(PulseInterfaces)
