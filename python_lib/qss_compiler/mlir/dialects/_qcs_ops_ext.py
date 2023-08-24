try:
    from ..ir import *
    from .qcs import *
    from .._mlir_libs._ibmDialectsQCS import *
    from ._ods_common import get_default_loc_context as _get_default_loc_context

    from typing import Any, List, Optional, Union
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

class ShotInitOp:
    def __init__(self, numShots, *, startNCOs=False, loc=None, ip=None):
        super().__init__(loc=loc, ip=ip)
        self.attributes["qcs.numShots"] = IntegerAttr.get(IntegerType.get_signless(32), numShots)
        self.attributes["quir.startNCOs"] = BoolAttr.get(startNCOs)

