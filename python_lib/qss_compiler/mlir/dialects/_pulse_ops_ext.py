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

try:
    from ..ir import (
        ArrayAttr,
        DenseElementsAttr,
        StringAttr,
        TypeAttr,
        FlatSymbolRefAttr,
        FunctionType,
    )
    from ..ir import *  # noqa: F401, F403
    from .pulse import *  # noqa: F401, F403
    from .quir import (  # noqa: F401, F403
        AngleType,
        ConstantOp as QUIRConstantOp,
    )
    from .._mlir_libs._qeDialectsPulse import *  # noqa: F401, F403
    from ._ods_common import (  # noqa: F401
        get_default_loc_context as _get_default_loc_context,
    )
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

from typing import Any, List, Optional, Union  # noqa: F401


class CallSequenceOp:
    def __init__(self, res, callee, operands_, *, loc=None, ip=None):
        super().__init__(res, FlatSymbolRefAttr.get(str(callee)), operands_)


class Frame_CreateOp:
    """Specialization for the Frame_Create op class."""

    def __init__(self, uid, *, amp=None, freq=None, phase=None, loc=None, ip=None):
        super().__init__(
            StringAttr.get(str(uid)),
            initial_amp=amp,
            initial_freq=freq,
            initial_phase=phase,
            loc=loc,
            ip=ip,
        )


class MixFrameOp:
    """Specialization for the MixFrame op class."""

    def __init__(
        self,
        port,
        frame,
        *,
        initial_amp=None,
        initial_freq=None,
        initial_phase=None,
        loc=None,
        ip=None
    ):
        super().__init__(
            port,
            StringAttr.get(str(frame)),
            initial_amp=initial_amp,
            initial_freq=initial_freq,
            initial_phase=initial_phase,
            loc=loc,
            ip=ip,
        )


class PlayOp:
    def __init__(self, target, wfr, *, angle=None, ssb=None, amplitude=None, loc=None, ip=None):
        super().__init__(
            target,
            wfr,
            angle=angle,
            ssb=ssb,
            amplitude=amplitude,
            loc=loc,
            ip=ip,
        )


class Port_CreateOp:
    def __init__(self, uid, *, loc=None, ip=None):
        super().__init__(StringAttr.get(str(uid)), loc=loc, ip=ip)


class SequenceOp:
    def __init__(self, sym_name, inputs, results, *, sym_visibility="public", loc=None, ip=None):
        super().__init__(
            StringAttr.get(str(sym_name)),
            TypeAttr.get(FunctionType.get(inputs=inputs, results=results)),
            sym_visibility=StringAttr.get(str(sym_visibility)),
            loc=loc,
            ip=ip,
        )

    def add_entry_block(self):
        self.body.blocks.append(
            *FunctionType(TypeAttr(self.attributes["function_type"]).value).inputs
        )

    @property
    def entry_block(self):
        return self.body.blocks[0]

    @property
    def arguments(self):
        return self.entry_block.arguments

    def pulse_args(self, args):
        args_attr = []
        for i in args:
            args_attr.append(StringAttr.get(i))
        arrayAttr = ArrayAttr.get(args_attr)
        self.attributes.__setitem__("pulse.args", arrayAttr)
        return


class Waveform_CreateOp:
    def __init__(self, samples_2d, *, loc=None, ip=None):
        super().__init__(DenseElementsAttr.get(samples_2d), loc=loc, ip=ip)
