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
from conans import ConanFile
import os
import platform
from conan.tools.cmake import CMake, CMakeDeps, CMakeToolchain, cmake_layout
import subprocess


class QasmConan(ConanFile):
    name = "qasm"
    version = "0.3.0"
    url = "https://github.com/openqasm/qe-qasm.git"
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "examples": [True, False]}
    default_options = {
        "shared": False,
        "examples": False,
        "gmp:shared": True,
        "mpc:shared": True,
        "mpfr:shared": True,
    }
    license = "Proprietary"
    author = "IBM Quantum development team"
    topics = ("Compiler", "Parser", "OpenQASM3")
    description = "Compiler for OpenQASM3 language."

    def source(self):
        token = os.environ.get("GITHUB_PAT")
        if token is not None:
            self.run(f"git clone https://{token}@github.com/openqasm/qe-qasm.git .")
        else:
            self.run("git clone git@github.com:openqasm/qe-qasm.git .")

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
