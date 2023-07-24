# (C) Copyright IBM 2023.
#
# This code is part of Qiskit.
#
# This code is licensed under the Apache License, Version 2.0 with LLVM
# Exceptions. You may obtain a copy of this license in the LICENSE.txt
# file in the root directory of this source tree.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
import os

from conans import ConanFile, CMake, tools
from conans.tools import load


# Get version from environment variable.
# Note: QSSC_VERSION must be set in the environment when exporting
# to the Conan cache (i.e. via conan export or conan create).
# https://docs.conan.io/en/1.53/reference/conanfile/attributes.html#version
def get_version():
    return os.environ.get("QSSC_VERSION", None)


class QSSCompilerConan(ConanFile):
    name = "qss-compiler"
    version = get_version()
    url = "https://github.com/qiskit/qss-compiler"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "pythonlib": [True, False]}
    default_options = {"shared": False, "pythonlib": True}
    license = "Proprietary"
    author = "IBM Quantum development team"
    topics = ("Compiler", "Scheduler", "OpenQASM3")
    description = "An LLVM- and MLIR-based Quantum compiler that consumes OpenQASM 3.0"
    generators = ["CMakeToolchain", "CMakeDeps"]
    exports_sources = "*"

    def requirements(self):
        tool_pkgs = ["llvm", "clang-tools-extra"]
        for req in self.conan_data["requirements"]:
            self.requires(req)

    def configure(self):
        if self.settings.os == "Macos":
            self.options["qasm"].shared = True

    def build_requirements(self):
        tool_pkgs = ["llvm", "clang-tools-extra"]
        # Add packages necessary for build.
        for req in self.conan_data["requirements"]:
            if any(req.startswith(tool + "/") for tool in tool_pkgs):
                self.tool_requires(req)

    def _configure_cmake(self):
        cmake = CMake(self, generator="Ninja")
        cmake.definitions["CMAKE_TOOLCHAIN_FILE"] = "conan_toolchain.cmake"
        cmake.definitions["CMAKE_EXPORT_COMPILE_COMMANDS"] = "ON"
        # linking in parallel on all CPUs may take up more memory than
        # available in a typical CI worker for debug builds.
        if self.settings.build_type == "Debug":
            cmake.definitions["LLVM_PARALLEL_LINK_JOBS"] = "2"
        return cmake

    def build(self):
        cmake = self._configure_cmake()
        cmake.verbose = True
        cmake.configure()
        cmake.build()

        if self.options.pythonlib:
            self.run(
            f"cd {self.build_folder}/python_lib \
                    && pip install -e .[test]"
            )

        if self.should_test:
            self.test(cmake)


    def package(self):
        cmake = self._configure_cmake()
        cmake.install()

        if self.options.pythonlib:
            self.run(
            f"cd {self.build_folder}/python_lib \
                    && pip install .[test]"
            )

        if self.should_test:
            self.test(cmake)

    def test(self, cmake):
        cmake = self._configure_cmake()
        cmake.test(target="check-tests")
        self.run(f"cd {self.source_folder} && pytest test/python_lib")

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
