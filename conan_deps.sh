#!/bin/bash

conan export ./conan/qasm qss/stable
conan export ./conan/llvm/conanfile.py @
conan export ./conan/clang-tools-extra/conanfile.py @
