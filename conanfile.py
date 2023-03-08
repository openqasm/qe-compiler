# (C) Copyright IBM 2023.
#
# This code is part of Qiskit.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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
    options = {"shared": [True, False]}
    default_options = {"shared": False}
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

    def package(self):
        cmake = self._configure_cmake()
        cmake.install()
        self.test()

    def test(self):
        cmake = self._configure_cmake()
        cmake.build(target="check-tests")
        cmake.build(target="check-format")
        self.run(
            f"cd {self.build_folder}/qss-compiler/python_lib \
                    && pip install .[test] --use-feature=in-tree-build"
        )
        self.run(f"cd {self.build_folder} && pytest qss-compiler/python_lib/test")

    def package_info(self):
        self.cpp_info.libs = tools.collect_libs(self)
