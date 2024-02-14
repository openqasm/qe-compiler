#!/usr/bin/env python
# coding: utf-8

# Using this notebook requires manual install of MLIR python bindings:
# 1. Copy or link to `qss-compiler/third_party/mlir/pyproject.toml` from `~/.conan/data/llvm/qss/stable/<hash>/python_packages/mlir_core`
# 2. Run `pip install -e .`
# 3. Copy or link to `~/.conan/data/llvm/qss/stable/<hash>/python_packages/mlir_core/mlir/_mlir_libs/libMLIRPythonCAPI.dylib` from `<python>/lib`
# 4. build compiler
# 5. cd `qss_compiler/python_lib`
# 6. run `bash setup_mlir.sh`
# 7. run `pip install -e .`

# In[1]:


from qss_compiler.mlir.ir import Context, InsertionPoint, Location, Module
from qss_compiler.mlir.ir import F64Type, IntegerType, IndexType
from qss_compiler.mlir.dialects import arith, builtin, std, scf, complex


# In[2]:


from qss_compiler.mlir.dialects import pulse, quir # noqa: F401, E402


# In[4]:


import numpy as np
import sys


# In[5]:



with Context() as ctx:
    
    pulse.pulse.register_dialect()
    quir.quir.register_dialect()

    with open(sys.argv[1]) as f:
        data = f.read()
    print(data)
    module = Module.parse(data)

        
            
print(str(module))


