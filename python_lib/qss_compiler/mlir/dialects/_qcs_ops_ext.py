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
    from ..ir import IntegerAttr, IntegerType, BoolAttr
    from .oq3 import *  # noqa: F401, F403
    from .._mlir_libs._qeDialectsOQ3 import *  # noqa: F401, F403
    from ._ods_common import (  # noqa: F401
        get_default_loc_context as _get_default_loc_context,
    )

    from typing import Any, List, Optional, Union  # noqa: F401
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e


class ShotInitOp:
    def __init__(self, numShots, *, startNCOs=False, loc=None, ip=None):
        super().__init__(loc=loc, ip=ip)
        self.attributes["qcs.numShots"] = IntegerAttr.get(
            IntegerType.get_signless(32), numShots
        )  # black/flake8 line length conflict
        self.attributes["quir.startNCOs"] = BoolAttr.get(startNCOs)
