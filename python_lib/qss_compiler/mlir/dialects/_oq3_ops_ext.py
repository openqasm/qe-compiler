try:
    from mlir.ir import *
    from .OQ3 import *
    from .._mlir_libs._ibmDialectsOQ3 import *
    from mlir.dialects._ods_common import get_default_loc_context as _get_default_loc_context

    from typing import Any, List, Optional, Union
except ImportError as e:
    raise RuntimeError("Error loading imports from extension module") from e



