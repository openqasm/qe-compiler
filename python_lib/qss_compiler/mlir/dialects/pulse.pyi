#  Based on Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# based on llvm-project/mlir/python/mlir/_mlir_libs/_mlir/dialects/pdl.pyi

from typing import Optional

from ..ir import Type, Context

__all__ = [
    "CaptureType",
    "FrameType",
    "KernelType",
    "PortType",
    "PortGroupType",
    "WaveformType",
    "MixedFrameType",
]

class CaptureType(Type):
    @staticmethod
    def isinstance(type: Type) -> bool: ...
    @staticmethod
    def get(context: Optional[Context] = None) -> CaptureType: ...

class FrameType(Type):
    @staticmethod
    def isinstance(type: Type) -> bool: ...
    @staticmethod
    def get(context: Optional[Context] = None) -> FrameType: ...

class KernelType(Type):
    @staticmethod
    def isinstance(type: Type) -> bool: ...
    @staticmethod
    def get(element_type: Type) -> KernelType: ...

    # @property
    # def element_type(self) -> Type: ...

class PortType(Type):
    @staticmethod
    def isinstance(type: Type) -> bool: ...
    @staticmethod
    def get(context: Optional[Context] = None) -> PortType: ...

class WaveformType(Type):
    @staticmethod
    def isinstance(type: Type) -> bool: ...
    @staticmethod
    def get(context: Optional[Context] = None) -> WaveformType: ...

class MixedFrameType(Type):
    @staticmethod
    def isinstance(type: Type) -> bool: ...
    @staticmethod
    def get(context: Optional[Context] = None) -> MixedFrameType: ...
