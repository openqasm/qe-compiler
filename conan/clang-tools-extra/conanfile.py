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
from conans.errors import ConanInvalidConfiguration
from conans import ConanFile, CMake, tools

import os.path
import os

LLVM_TAG = "llvmorg-17.0.5"


class ClangToolsExtraConan(ConanFile):
    name = "clang-tools-extra"
    version = "17.0.5-0"
    description = "A toolkit for analysis of c++ projects."
    license = "Apache-2.0 WITH LLVM-exception"
    topics = ("conan", "llvm", "clang-tools-extra")
    homepage = "https://github.com/llvm/llvm-project/tree/master/llvm"
    url = "https://github.com/conan-io/conan-center-index"

    settings = ("os", "arch", "compiler", "build_type")
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
    }

    generators = ["cmake"]
    no_copy_source = True
    exports_sources = "llvm-project/*"

    def source(self):
        git_cache = os.environ.get("CONAN_LLVM_GIT_CACHE")
        cache_hit = os.path.exists(f"{git_cache}/.git")
        cache_arg = f" --reference-if-able '{git_cache}' " if git_cache else ""

        if git_cache and cache_hit:
            self.output.info(f"Cache hit! Some Git objects will be loaded from '{git_cache}'.")

        self.run(
            f"git clone {cache_arg} -b {LLVM_TAG} "
            "--single-branch https://github.com/llvm/llvm-project.git"
        )

        if git_cache and not cache_hit:
            # Update cache.
            self.output.info(f"Updating cache at '{git_cache}'.")
            self.run(f"cp -r llvm-project '{git_cache}'")

    @property
    def _source_subfolder(self):
        return "llvm-project/llvm"

    def _supports_compiler(self):
        compiler = self.settings.compiler.value
        version = tools.Version(self.settings.compiler.version)
        major_rev, _ = int(version.major), int(version.minor)

        unsupported_combinations = [
            [compiler == "gcc", major_rev < 8],
            [compiler == "clang", major_rev < 10],
            [compiler == "apple-clang", major_rev < 11],
        ]
        if any(all(combination) for combination in unsupported_combinations):
            message = 'unsupported compiler: "{}", version "{}"'
            raise ConanInvalidConfiguration(message.format(compiler, version))

    def _configure_cmake(self):
        cmake = CMake(self, generator="Ninja")
        cmake.definitions["LLVM_ENABLE_PROJECTS"] = "clang;clang-tools-extra"
        cmake.definitions["BUILD_SHARED_LIBS"] = False
        cmake.definitions["CMAKE_SKIP_RPATH"] = True
        cmake.definitions["CMAKE_BUILD_TYPE"] = "Release"
        cmake.definitions["CMAKE_POSITION_INDEPENDENT_CODE"] = (
            self.options.get_safe("fPIC", default=False) or self.options.shared
        )

        cmake.definitions["LLVM_TARGET_ARCH"] = "host"
        cmake.definitions["LLVM_TARGETS_TO_BUILD"] = ""
        cmake.definitions["LLVM_ENABLE_PIC"] = self.options.get_safe("fPIC", default=False)

        cmake.definitions["LLVM_ABI_BREAKING_CHECKS"] = "WITH_ASSERTS"
        cmake.definitions["LLVM_ENABLE_WARNINGS"] = False
        cmake.definitions["LLVM_ENABLE_PEDANTIC"] = True
        cmake.definitions["LLVM_ENABLE_WERROR"] = False

        cmake.definitions["LLVM_USE_RELATIVE_PATHS_IN_DEBUG_INFO"] = False
        cmake.definitions["LLVM_BUILD_INSTRUMENTED_COVERAGE"] = False
        cmake.definitions["LLVM_REVERSE_ITERATION"] = False
        cmake.definitions["LLVM_ENABLE_BINDINGS"] = False
        cmake.definitions["LLVM_CCACHE_BUILD"] = False

        cmake.definitions["LLVM_INCLUDE_EXAMPLES"] = False
        cmake.definitions["LLVM_INCLUDE_TESTS"] = False
        cmake.definitions["LLVM_INCLUDE_BENCHMARKS"] = False
        cmake.definitions["LLVM_APPEND_VC_REV"] = False
        cmake.definitions["LLVM_BUILD_DOCS"] = False
        cmake.definitions["LLVM_ENABLE_IDE"] = False

        cmake.definitions["LLVM_ENABLE_EH"] = True
        cmake.definitions["LLVM_ENABLE_RTTI"] = True
        cmake.definitions["LLVM_ENABLE_THREADS"] = False
        cmake.definitions["LLVM_ENABLE_LTO"] = False
        cmake.definitions["LLVM_STATIC_LINK_CXX_STDLIB"] = False
        cmake.definitions["LLVM_ENABLE_UNWIND_TABLES"] = False
        cmake.definitions["LLVM_ENABLE_EXPENSIVE_CHECKS"] = False
        cmake.definitions["LLVM_ENABLE_ASSERTIONS"] = False
        cmake.definitions["LLVM_USE_NEWPM"] = False
        cmake.definitions["LLVM_USE_OPROFILE"] = False
        cmake.definitions["LLVM_USE_PERF"] = False
        cmake.definitions["LLVM_USE_SANITIZER"] = ""
        cmake.definitions["LLVM_ENABLE_Z3_SOLVER"] = False
        cmake.definitions["LLVM_ENABLE_LIBPFM"] = False
        cmake.definitions["LLVM_ENABLE_LIBEDIT"] = False
        cmake.definitions["LLVM_ENABLE_FFI"] = False
        cmake.definitions["LLVM_ENABLE_ZLIB"] = False
        cmake.definitions["LLVM_ENABLE_LIBXML2"] = False
        return cmake

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        self._supports_compiler()

    def build(self):
        cmake = self._configure_cmake()
        cmake.configure(source_folder=self._source_subfolder)
        cmake.build()

    def package(self):
        cmake = self._configure_cmake()
        cmake.install()
        lib_path = os.path.join(self.package_folder, "lib")
        share_path = os.path.join(self.package_folder, "share/clang")
        bin_path = os.path.join(self.package_folder, "bin")

        patterns = ["tidy", "format", "apply"]
        if os.path.exists(bin_path):
            for name in os.listdir(bin_path):
                if not any(pattern in name for pattern in patterns):
                    os.remove(os.path.join(bin_path, name))

        if os.path.exists(lib_path):
            for name in os.listdir(lib_path):
                file = os.path.join(lib_path, name)
                if not os.path.isdir(file):
                    os.remove(file)

        self.copy("*.py", dst="bin", src=share_path)
        tools.rmdir(os.path.join(self.package_folder, "include"))
        tools.rmdir(os.path.join(self.package_folder, "share"))
        tools.rmdir(os.path.join(lib_path, "cmake"))

    def package_id(self):
        self.info.include_build_settings()
        del self.info.settings.build_type
