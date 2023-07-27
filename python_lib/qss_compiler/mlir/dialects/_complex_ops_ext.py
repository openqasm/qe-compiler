try:
    from mlir.ir import *
    from .complex import *
    from mlir.dialects._ods_common import get_default_loc_context as _get_default_loc_context

    from typing import Any, List, Optional, Union
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e


class CreateOp:
    def __init__(self, real, imaginary, *, loc=None, ip=None):
        super().__init__(ComplexType.get(F64Type.get()), real, imaginary, loc=None, ip=None)
