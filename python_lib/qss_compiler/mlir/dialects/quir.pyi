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

# based on llvm-project/mlir/python/mlir/_mlir_libs/_mlir/dialects/pdl.pyi

from typing import Optional

from ..ir import Type, Context

__all__ = [
    "AngleType",
]

class AngleType(Type):
    @staticmethod
    def isinstance(type: Type) -> bool: ...
    @staticmethod
    def get(context: Optional[Context] = None) -> AngleType: ...
