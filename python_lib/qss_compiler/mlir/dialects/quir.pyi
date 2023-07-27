#  Based on Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# based on llvm-project/mlir/python/mlir/_mlir_libs/_mlir/dialects/pdl.pyi

from typing import Optional

from mlir.ir import Type, Context

__all__ = [
    "AngleType",
]

class AngleType(Type):
    @staticmethod
    def isinstance(type: Type) -> bool: ...
    @staticmethod
    def get(context: Optional[Context] = None) -> AngleType: ...
