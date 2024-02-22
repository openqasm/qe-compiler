try:
    from ..ir import *  # noqa: F401, F403
    from ..ir import IntegerAttr, IntegerType, BoolAttr
    from .oq3 import *  # noqa: F401, F403
    from .._mlir_libs._ibmDialectsOQ3 import *  # noqa: F401, F403
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
