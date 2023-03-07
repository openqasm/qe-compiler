#!/bin/bash
#
# (C) Copyright IBM 2023.
#
# This code is part of Qiskit.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific

conan export ./conan/qasm qss/stable
conan export ./conan/llvm/conanfile.py @
conan export ./conan/clang-tools-extra/conanfile.py @
