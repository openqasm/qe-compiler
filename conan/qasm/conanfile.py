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
from conans import ConanFile, tools
import os, platform
from contextlib import contextmanager
from conans.tools import load
from conan.tools.apple import is_apple_os
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
import subprocess


class QasmConan(ConanFile):
    name = "qasm"
    version = "0.2.13"
    url = "https://github.com/Qiskit/qss-qasm.git"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "examples": [True, False]}
    default_options = {"shared": False, "examples": False}
    license = "Proprietary"
    author = "IBM Quantum development team"
    topics = ("Compiler", "Parser", "OpenQASM3")
    description = "Compiler for OpenQASM3 language."

    def source(self):
        token = os.environ.get("GITHUB_PAT")
        if token is not None:
            self.run(f"git clone https://{token}@github.com/Qiskit/qss-qasm.git .")
        else:
            self.run(f"git clone git@github.com:Qiskit/qss-qasm.git .")

        commit_hash = self.conan_data["sources"]["hash"]
        self.run(f"git checkout {commit_hash}")

    def requirements(self):
        # Private deps won't be linked against by consumers, which is important
        # at least for Flex which does not expose a CMake target.
        private_deps = ["bison", "flex"]
        for req in self.conan_data["requirements"]:
            private = any(req.startswith(d) for d in private_deps)
            self.requires(req, private=private)

    def build_requirements(self):
        for req in self.conan_data["build_requirements"]:
            self.tool_requires(req)

    def generate(self):
        tc = CMakeToolchain(self)
        tc.cache_variables["BUILD_SHARED_LIBS"] = self.options.shared
        tc.cache_variables["BUILD_STATIC_LIBS"] = not self.options.shared
        tc.cache_variables["OPENQASM_BUILD_EXAMPLES"] = self.options.examples
        tc.generate()

        deps = CMakeDeps(self)
        deps.generate()

    def layout(self):
        cmake_layout(self)

    def build(self):
        cmake = CMake(self)
        cmake.configure()

        # Note that if a job does not produce output for a longer period of
        # time, then Travis will cancel that job.
        # Avoid that timeout by running vmstat in the background, which reports
        # free memory and other stats every 30 seconds.
        use_monitor = platform.system() == "Linux"
        if use_monitor:
            subprocess.run("free -h ; lscpu", shell=True)
            monitor = subprocess.Popen(["vmstat", "-w", "30", "-t"])
        cmake.build()
        if use_monitor:
            monitor.terminate()
            monitor.wait(1)

    def package(self):
        cmake = CMake(self)
        cmake.install()
        self.copy("*.tab.cpp", dst="include/lib/Parser", src="lib/Parser")
        self.copy("*.tab.h", dst="include/lib/Parser", src="lib/Parser")

    def package_info(self):
        self.cpp_info.set_property("cmake_find_mode", "both")
        self.cpp_info.set_property("cmake_file_name", "qasm")
        self.cpp_info.set_property("cmake_target_name", "qasm::qasm")

        self.cpp_info.libs = ["qasmParser", "qasmFrontend", "qasmAST", "qasmDIAG"]
