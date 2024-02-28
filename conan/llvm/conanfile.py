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
import sys


LLVM_TAG = "llvmorg-17.0.5"


class LLVMConan(ConanFile):
    name = "llvm"
    version = "17.0.5-0"
    description = (
        "A toolkit for the construction of highly optimized compilers,"
        "optimizers, and runtime environments."
    )
    license = "Apache-2.0 WITH LLVM-exception"
    topics = ("conan", "llvm")
    homepage = "https://github.com/llvm/llvm-project/tree/master/llvm"
    url = "https://github.com/conan-io/conan-center-index"

    settings = ("os", "arch", "compiler", "build_type")
    options = {
        "shared": [True, False],
        "fPIC": [True, False],
        "components": "ANY",
        "targets": "ANY",
        "exceptions": [True, False],
        "external_lit": "ANY",
        "rtti": [True, False],
        "threads": [True, False],
        "lto": ["On", "Off", "Full", "Thin"],
        "static_stdlib": [True, False],
        "unwind_tables": [True, False],
        "expensive_checks": [True, False],
        "use_perf": [True, False],
        "use_sanitizer": [
            "Address",
            "Memory",
            "MemoryWithOrigins",
            "Undefined",
            "Thread",
            "DataFlow",
            "Address;Undefined",
            "None",
        ],
        "with_ffi": [True, False],
        "with_zlib": [True, False],
        "with_xml2": [True, False],
        "with_python_bindings": [True, False],
    }
    default_options = {
        "shared": False,
        "fPIC": True,
        "components": "all",
        "targets": "X86;PowerPC",
        "exceptions": True,
        "external_lit": "None",
        "rtti": True,
        "threads": True,
        "lto": "Off",
        "static_stdlib": False,
        "unwind_tables": True,
        "expensive_checks": False,
        "use_perf": False,
        "use_sanitizer": "None",
        "with_ffi": False,
        "with_zlib": True,
        "with_xml2": False,
        "with_python_bindings": True,
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

    def _patch_build(self):
        if os.path.exists("FindIconv.cmake"):
            tools.replace_in_file("FindIconv.cmake", "iconv charset", "iconv")

    def _configure_cmake(self):
        cmake = CMake(self, generator="Ninja")
        cmake.definitions["LLVM_ENABLE_PROJECTS"] = os.environ.get(
            "LLVM_ENABLE_PROJECTS", "mlir;lld"
        )
        cmake.definitions["BUILD_SHARED_LIBS"] = False
        cmake.definitions["CMAKE_SKIP_RPATH"] = True
        cmake.definitions["CMAKE_POSITION_INDEPENDENT_CODE"] = (
            self.options.get_safe("fPIC", default=False) or self.options.shared
        )

        if not self.options.shared:
            cmake.definitions["DISABLE_LLVM_LINK_LLVM_DYLIB"] = True

        cmake.definitions["LLVM_TARGET_ARCH"] = "host"
        cmake.definitions["LLVM_TARGETS_TO_BUILD"] = self.options.targets
        cmake.definitions["LLVM_BUILD_LLVM_DYLIB"] = self.options.shared
        cmake.definitions["LLVM_DYLIB_COMPONENTS"] = self.options.components
        cmake.definitions["LLVM_ENABLE_PIC"] = self.options.get_safe("fPIC", default=False)

        cmake.definitions["LLVM_ABI_BREAKING_CHECKS"] = "WITH_ASSERTS"
        cmake.definitions["LLVM_ENABLE_WARNINGS"] = False
        cmake.definitions["LLVM_ENABLE_PEDANTIC"] = True
        cmake.definitions["LLVM_ENABLE_WERROR"] = False

        if self.options.external_lit != "None":
            cmake.definitions["LLVM_EXTERNAL_LIT"] = self.options.external_lit

        cmake.definitions["LLVM_TEMPORARILY_ALLOW_OLD_TOOLCHAIN"] = True
        cmake.definitions["LLVM_USE_RELATIVE_PATHS_IN_DEBUG_INFO"] = False
        cmake.definitions["LLVM_BUILD_INSTRUMENTED_COVERAGE"] = False
        cmake.definitions["LLVM_OPTIMIZED_TABLEGEN"] = True
        cmake.definitions["LLVM_REVERSE_ITERATION"] = False
        cmake.definitions["LLVM_ENABLE_BINDINGS"] = False
        cmake.definitions["LLVM_CCACHE_BUILD"] = False

        cmake.definitions["LLVM_BUILD_TOOLS"] = True
        cmake.definitions["LLVM_INCLUDE_TOOLS"] = True

        cmake.definitions["LLVM_INSTALL_UTILS"] = True
        cmake.definitions["LLVM_INCLUDE_EXAMPLES"] = False
        cmake.definitions["LLVM_INCLUDE_TESTS"] = False
        cmake.definitions["LLVM_INCLUDE_BENCHMARKS"] = False
        cmake.definitions["LLVM_APPEND_VC_REV"] = False
        cmake.definitions["LLVM_BUILD_DOCS"] = False
        cmake.definitions["LLVM_ENABLE_IDE"] = False

        cmake.definitions["LLVM_ENABLE_EH"] = self.options.exceptions
        cmake.definitions["LLVM_ENABLE_RTTI"] = self.options.rtti
        cmake.definitions["LLVM_ENABLE_THREADS"] = self.options.threads
        cmake.definitions["LLVM_ENABLE_LTO"] = self.options.lto
        cmake.definitions["LLVM_STATIC_LINK_CXX_STDLIB"] = self.options.static_stdlib
        cmake.definitions["LLVM_ENABLE_UNWIND_TABLES"] = self.options.unwind_tables
        cmake.definitions["LLVM_ENABLE_EXPENSIVE_CHECKS"] = self.options.expensive_checks
        cmake.definitions["LLVM_ENABLE_ASSERTIONS"] = False
        #            self.settings.build_type == 'Debug'

        cmake.definitions["LLVM_ENABLE_TERMINFO"] = False

        cmake.definitions["LLVM_USE_NEWPM"] = False
        cmake.definitions["LLVM_USE_OPROFILE"] = False
        cmake.definitions["LLVM_USE_PERF"] = self.options.use_perf
        if self.options.use_sanitizer == "None":
            cmake.definitions["LLVM_USE_SANITIZER"] = ""
        else:
            cmake.definitions["LLVM_USE_SANITIZER"] = self.options.use_sanitizer

        cmake.definitions["LLVM_ENABLE_Z3_SOLVER"] = False
        cmake.definitions["LLVM_ENABLE_LIBPFM"] = False
        cmake.definitions["LLVM_ENABLE_LIBEDIT"] = False
        cmake.definitions["LLVM_ENABLE_FFI"] = self.options.with_ffi
        cmake.definitions["LLVM_ENABLE_ZLIB"] = self.options.get_safe("with_zlib", False)
        cmake.definitions["LLVM_ENABLE_LIBXML2"] = self.options.get_safe("with_xml2", False)

        cmake.definitions["LLVM_PARALLEL_LINK_JOBS"] = 4

        cmake.definitions["MLIR_ENABLE_BINDINGS_PYTHON"] = self.options.get_safe(
            "with_python_bindings", False
        )
        cmake.definitions["Python3_EXECUTABLE"] = sys.executable

        if self.settings.build_type == "Debug":
            cmake.definitions["CMAKE_C_FLAGS"] = "-gz=zlib"
            cmake.definitions["CMAKE_CXX_FLAGS"] = "-gz=zlib"
            cmake.definitions["CMAKE_EXE_LINKER_FLAGS"] = "-gz=zlib"

        return cmake

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC
            del self.options.with_zlib
            del self.options.with_xml2

    def requirements(self):
        if self.options.with_ffi:
            self.requires("libffi/3.3")
        if self.options.get_safe("with_zlib", False):
            self.requires("zlib/1.2.13")
        if self.options.get_safe("with_xml2", False):
            self.requires("libxml2/2.9.10")

    def configure(self):
        if self.options.shared:  # Shared builds disabled just due to the CI
            message = "Shared builds not currently supported"
            raise ConanInvalidConfiguration(message)
            # del self.options.fPIC
        # if self.settings.os == 'Windows' and self.options.shared:
        #     message = 'Shared builds not supported on Windows'
        #     raise ConanInvalidConfiguration(message)
        if self.options.exceptions and not self.options.rtti:
            message = "Cannot enable exceptions without rtti support"
            raise ConanInvalidConfiguration(message)
        if tools.is_apple_os(self.settings.os) and self.settings.arch == "armv8":
            self.options.targets = self.options.targets.value + ";AArch64"

        self._supports_compiler()

    def build(self):
        self._patch_build()
        cmake = self._configure_cmake()
        cmake.configure(source_folder=self._source_subfolder)
        cmake.build()

    def package(self):
        cmake = self._configure_cmake()
        cmake.install()
        self.copy("llvm-lit", dst="bin", src="bin")

    def package_info(self):
        # Don't generate any find modules. LLVM provides its own
        # in lib/cmake, which will get picked up via builddirs.
        self.cpp_info.set_property("cmake_find_mode", "none")
        self.cpp_info.builddirs = ["lib/cmake"]

        # TODO: remove once we've switched to Conan 2.0
        self.cpp_info.names["cmake_find_package"] = "LLVM"
        self.cpp_info.names["cmake_find_package_multi"] = "LLVM"

        lib_path = os.path.join(self.package_folder, "lib")
        skip_words = [".dylib", ".so", "cmake", "objects-"]
        for name in os.listdir(lib_path):
            if not any(skip_word in name for skip_word in skip_words):
                self.cpp_info.libs.append(name)
            else:
                self.output.info(f"Skipping library: {name}")
