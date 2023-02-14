# (C) Copyright IBM 2021, 2023.
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
    options = {"shared": [True, False]}
    default_options = {"shared": False}
    license = "Proprietary"
    author = "IBM Quantum development team"
    topics = ("Compiler", "Scheduler", "OpenQASM3")
    description = "An LLVM- and MLIR-based Quantum compiler that consumes OpenQASM 3.0"
    generators = ["cmake", "cmake_find_package"]
    exports_sources = "*"

    def requirements(self):
        for req in self.conan_data["requirements"]:
            self.requires(req)

    def _configure_cmake(self):
        cmake = CMake(self, generator="Ninja")
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
