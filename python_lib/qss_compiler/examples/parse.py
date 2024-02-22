#!/usr/bin/env python
# coding: utf-8

# Using this notebook requires manual install of MLIR python bindings:
# 1. Copy or link to `qss-compiler/third_party/mlir/pyproject.toml` from `~/.conan/data/llvm/qss/stable/<hash>/python_packages/mlir_core` # noqa: E501
# 2. Run `pip install -e .`
# 3. Copy or link to `~/.conan/data/llvm/qss/stable/<hash>/python_packages/mlir_core/mlir/_mlir_libs/libMLIRPythonCAPI.dylib` from `<python>/lib`  # noqa: E501
# 4. build compiler
# 5. cd `qss_compiler/python_lib`
# 6. run `bash setup_mlir.sh`
# 7. run `pip install -e .`

# In[1]:


import sys
from qss_compiler.mlir.ir import Context, Module
from qss_compiler.mlir.dialects import pulse, quir

# In[2]:


with Context() as ctx:

    pulse.pulse.register_dialect()
    quir.quir.register_dialect()

    with open(sys.argv[1]) as f:
        data = f.read()
    print(data)
    module = Module.parse(data)


print(str(module))
