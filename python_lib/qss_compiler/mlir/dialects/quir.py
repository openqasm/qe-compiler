from ._quir_ops_gen import *  # noqa: F403, F401
from ._mlir_libs._ibmDialectsQUIR import *  # noqa: F403, F401

from mlir.ir import UnitAttr
from mlir.dialects import scf


class ShotLoop(scf.ForOp):
    def __init__(self, lb, ub, step):
        super().__init__(lb, ub, step)
        self.attributes["quir.shotLoop"] = UnitAttr.get()
