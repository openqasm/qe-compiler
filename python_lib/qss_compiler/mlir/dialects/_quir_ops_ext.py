try:
    from mlir.ir import *
    from .quir import *
    from ._mlir_libs._ibmDialectsQUIR import *
    from mlir.dialects._ods_common import get_default_loc_context as _get_default_loc_context

    from typing import Any, List, Optional, Union
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e

class ConstantOp:
    def __init__(self, constant_type, value, *, loc=None, ip=None):
        if constant_type == "angle":
            result = AngleType.get(20)
            value = Attribute.parse(f"#quir.angle<{value}: !quir.angle<20>>")
        elif constant_type == "duration":
            result = DurationType.get()
            value = Attribute.parse(f'#quir.duration<"{value}" : !quir.duration>')
        else:
            raise ValueError(f"Unknown quir constant type {constant_type}")

        super().__init__(result, value, loc=loc, ip=ip)


class DelayOp:
    # TODO: Why is target and duration apparently swapped?
    def __init__(self, target, *, dur=[], loc=None, ip=None):
        super().__init__(target, dur, loc=loc, ip=ip)
