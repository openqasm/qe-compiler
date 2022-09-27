#!/usr/bin/env python3
#
# (C) Copyright IBM 2021.
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
