#!/bin/bash

conan export ./conan/qasm qss/stable
conan export ./conan/llvm/conanfile_llvm.py qss/stable
conan export ./conan/llvm/conanfile_clang-tools-extra.py qss/stable
