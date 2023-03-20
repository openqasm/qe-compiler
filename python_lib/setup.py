#!/usr/bin/env python3
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


"""Setup module for Quantum System Software Compiler package."""

from setuptools import setup, Extension

setup(
    # This distribution contains platform-specific C++ libraries, but they are not
    # built with distutils. We create a dummy Extension object so that distutils
    # knows to make the binary platform-specific.
    ext_modules=[Extension("dummy", sources=["dummy.c"])],
)
