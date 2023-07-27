#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._complex_ops_gen import *  # noqa: F403, F401

# manually copied from github
# https://github.com/llvm/llvm-project/blob/main/mlir/python/mlir/dialects/complex.py
# 11/29/2022

# the complex dialect does not appear to be available in llvm 14.0
