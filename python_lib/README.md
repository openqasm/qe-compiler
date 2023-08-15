# Python API for the [QSS Compiler](../README.md)

This `python_lib` subdirectory of the
[qss-compiler](https://github.com/Qiskit/qss-compiler/) provides python
bindings for the qss-compiler API. In particular, it creates a python
package called `qss_compiler` that exposes two functions that compile input from a string or from a file, respectively:

```
compile_str(input_str: str,
             compile_options: Optional[CompileOptions],
             **kwargs,
) -> Union[bytes, str, None]

compile_file(input_file: str,
             compile_options: Optional[CompileOptions],
             **kwargs,
) -> Union[bytes, str, None]`
```

Both support delivering the compiler's output by returning a byte sequence (i.e., python's `bytes`) or by writing to a file. See the tests in `test/python_lib/test_compile.py` for examples of the supported combinations of input and output (note that the combinations in these examples are not exhaustive).

For generating payloads (`OutputType.QEM`), the functions must be called with a valid `target` and path to configuration (`config_path`), which is also demonstrated in the tests.


## Installation

### From source

To install the package in the current virtual environment, from `<build dir>/python_lib`, run:

    pip install .

For development, we recommend a so-called _editable_ install that links back to the work tree.

    pip install -e .

Note that when you change python or C++ source files, you need to first re-build `qss-compiler` (i.e., run `make` or `ninja`) so that the changes become visible in the build directory, before rebuilding or reinstalling the python package.

The build system copies over python source files and all other files required to building the python package to `<build dir>/python_lib`. Also, the build system generates a shared library file containing the QSSC python module. Specifically, you should see a file in `<build dir>/python_lib/qss_compiler/` named like `py_qssc.cpython-<pyversion>-<arch>.so` (e.g. `py_qssc.cpython-39-darwin.so`).
The reason for that indirection is that python cannot collect files from different directories when building a package. Thus, the generated shared library and the python source files from the source tree need to be moved to the same directory hierarchy.


## Testing

From the `qss-compiler` directory, after installation, simply run:

    pytest test/python_lib