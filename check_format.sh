#!/bin/bash
#
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


# Expects first argument as path to clang-format binary
# following arguments as paths to files to format

clang_format_bin=$1
files="${@:2}"

diff <(cat ${files}) <(${clang_format_bin} --style=file ${files})
