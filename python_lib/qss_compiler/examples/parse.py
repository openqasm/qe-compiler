#!/usr/bin/env python
# coding: utf-8

# (C) Copyright IBM 2024.
#
# This code is part of Qiskit.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from qss_compiler.mlir.ir import Context, Module
from qss_compiler.mlir.dialects import pulse, quir

with Context() as ctx:

    pulse.pulse.register_dialect()
    quir.quir.register_dialect()

    with open(sys.argv[1]) as f:
        data = f.read()
    print(data)
    module = Module.parse(data)


print(str(module))
