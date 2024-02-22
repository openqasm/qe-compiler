try:
    from ..ir import *  # noqa: F401, F403
    from ..ir import Attribute
    from .oq3 import *  # noqa: F401, F403
    from .._mlir_libs._ibmDialectsOQ3 import *  # noqa: F401, F403
    from .._mlir_libs._ibmDialectsQUIR import (
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
