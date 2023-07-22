
# qss-compiler: An MLIR based quantum compiler for control systems

The qss-compiler is an MLIR based quantum control system compiler with support for OpenQASM 3. It is designed to compiler quantum programs to real quantum hardware and is designed as part the overall Quantum Engine.

## Building
We use [Conan](https://docs.conan.io/en/1.59/index.html) to build the compiler and handle dependencies.

To build:

1. `git clone git@github.com:Qiskit/qss-compiler.git && cd qss-compiler`
2. Install Python dependencies: `pip install -r requirements-dev.txt`
3. Export local Conan recipe dependencies to Conan: `./conan_deps.sh`
4. `mkdir build && cd build`
5. Install Conan dependencies: `conan install .. --build=outdated -pr:h default -pr:b default`
6. Invoke the build with Conan: `conan build ..`. This will build the compiler.
7. To run checks: `ninja check-tests`

Alternatively to step 6, you can build directly with CMake (also from within the build folder):
```
cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE="Release" ..
ninja
```
