
# qe-compiler: An MLIR-based quantum compiler for quantum engines

The qe-compiler is an [MLIR](https://mlir.llvm.org/)-based compiler with support for OpenQASM 3. It is designed to compile quantum programs to quantum hardware and is designed as part of the overall Quantum Engine. This repo contains the compiler front-end to convert OpenQASM 3 source files into a collection of four MLIR dialects called QUIR (QUantum Intermediate Representation), OQ3 (OpenQASM 3), Pulse (OpenPulse), and QCS (Quantum Computing System). This set of dialects allows OpenQASM programs to be converted into a form suitable to manipulate with LLVM. This repo also contains tools and compiler passes that are agnostic of details of any control system vendor. For instance, it contains localization passes to split source programs into the qubit or channel-specific groupings required by a target quantum control system.

This repo does not contain a complete compiler. Rather, it is a framework for building compilers. To produce a complete compiler, one needs to implement at qe-compiler **target**. This repo comes with a "mock" target to assist developers in understanding how to develop such targets.

## Notice

We are in the process of changing the name of this project from `qss-compiler` to `qe-compiler`. At present, only the repository name and this README have been updated.

This open-source version of the qe-compiler is currently lacking documentation. We will add getting started guides and other resources in the near future.

## Building
We use [Conan](https://docs.conan.io/en/1.59/index.html) to build the compiler and handle dependencies.

To build:

1. `git clone git@github.com:Qiskit/qe-compiler.git && cd qe-compiler`
2. Install Python dependencies: `pip install -r requirements-dev.txt`
3. Export local Conan recipe dependencies to Conan: `./conan_deps.sh`
4. `mkdir build && cd build`
5. Install Conan dependencies: `conan install .. --build=outdated -pr:h default -pr:b default`
6. Invoke the build with Conan: `conan build ..`. This will build the compiler.
7. To run tests: `conan build .. --test`

Alternatively instead of steps 6/7, you can build directly with CMake (also from within the build folder):
1. Configure - `cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE="Release" ..`
2. Build - `ninja`
3. Check tests - `ninja check-tests`

### Python library
The `qss-compiler` Python library will be installed by default to the resolved environment Python when
installing with conan. To disable add the option `conan build .. -o pythonlib=False`.

### Platforms
#### Windows
Building and running the compiler is supported with [WSL](https://learn.microsoft.com/en-us/windows/wsl/install).

## Example Use

You can inspect the options available when invoking the `qe-compiler` by passing the `-h` flag. There are **many** options. A basic invocation that will work without implementing a target will convert an OpenQASM source file, `example.qasm` to the MLIR dialect set described above with:
`qss-compiler --emit=mlir example.qasm`

## License
The qe-compiler is [licensed](LICENSE.txt) under the Apache License v2 with LLVM Exceptions.
