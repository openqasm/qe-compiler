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
    from ..ir import *  # noqa: F401, F403
    from ..ir import Attribute
    from .oq3 import *  # noqa: F401, F403
    from .._mlir_libs._qeDialectsOQ3 import *  # noqa: F401, F403
    from .._mlir_libs._qeDialectsQUIR import (
        AngleType,
        DurationType,
    )
    from ._ods_common import (  # noqa: F401
        get_default_loc_context as _get_default_loc_context,
    )
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e


class ConstantOp:
    def __init__(self, constant_type, value, *, loc=None, ip=None):
        if constant_type == "angle":
            result = AngleType.get(20)
            value = Attribute.parse(f"#quir.angle<{value}: !quir.angle<20>>")
        elif constant_type == "duration":
            result = DurationType.get()
            value = Attribute.parse(
                f'#quir.duration<"{value}" : !quir.duration>'
            )  # black/flake dispute about line length
        else:
            raise ValueError(f"Unknown quir constant type {constant_type}")

        super().__init__(result, value, loc=loc, ip=ip)


class DelayOp:
    # TODO: Why is target and duration apparently swapped?
    def __init__(self, target, *, dur=[], loc=None, ip=None):
        super().__init__(target, dur, loc=loc, ip=ip)
