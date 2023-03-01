from conans import ConanFile, CMake, tools
import os, platform
from contextlib import contextmanager
from conans.tools import load
from conan.tools.apple import is_apple_os

import subprocess


class QasmConan(ConanFile):
    name = 'qasm'
    version = "0.2.12"
    url = 'https://github.com/Qiskit/qss-qasm.git'
    settings = "os", "compiler", "build_type", "arch"
    options = {"shared": [True, False], "examples": [True, False]}
    default_options = {"shared": False, "examples": False}
    license = "Proprietary"
    author = "IBM Quantum development team"
    topics = ("Compiler", "Parser", "OpenQASM3")
    description = "Compiler for OpenQASM3 language."
    generators = "cmake"

    def source(self):
        self.run(f"git clone git@github.com:Qiskit/qss-qasm.git")

        commit_hash = self.conan_data["sources"]["hash"]
        self.run(f"cd qss-qasm && git checkout {commit_hash} && cd ..")

        # When building on Apple, we need to pass the KEEP_RPATHS option to the
        # basic setup command for Conan.
        keep_rpaths = "KEEP_RPATHS" if tools.is_apple_os(self.settings.os) else ""

        tools.replace_in_file("qss-qasm/CMakeLists.txt",
        '''project(OPENQASM VERSION "${OPENQASM_VERSION_TRIPLE}")''',
        '''project(OPENQASM VERSION "${OPENQASM_VERSION_TRIPLE}")
        if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
            message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
            file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/master/conan.cmake"
                "${CMAKE_BINARY_DIR}/conan.cmake")
        endif()
        include(${CMAKE_BINARY_DIR}/conan.cmake)
        conan_cmake_run(CONANFILE ${CMAKE_SOURCE_DIR}/conanfile.py
            BASIC_SETUP ''' + keep_rpaths + '''
            BUILD missing)
            ''')

        tools.replace_in_file("qss-qasm/lib/Parser/CMakeLists.txt",
        '''set(BISON_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/QasmParser.tab.cpp")''',
        '''set(SETUP_M4 ${CMAKE_COMMAND} -E env M4=${CONAN_BIN_DIRS_M4}/m4)
        set(BISON_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/QasmParser.tab.cpp")
        ''')

        tools.replace_in_file("qss-qasm/lib/Parser/CMakeLists.txt",
        '''COMMAND ${BISON_EXECUTABLE}''',
        '''COMMAND ${SETUP_M4} ${BISON_EXECUTABLE}
        ''')

    def requirements(self):
        for req in self.conan_data["requirements"]:
            self.requires(req)

    def build_requirements(self):
        for req in self.conan_data["build_requirements"]:
            self.tool_requires(req)

    def _configure_cmake(self):
        cmake = CMake(self, parallel=False)
        cmake.verbose = True
        cmake.definitions["BUILD_SHARED_LIBS"] = self.options.shared
        cmake.definitions["BUILD_STATIC_LIBS"] = not self.options.shared
        cmake.definitions["OPENQASM_BUILD_EXAMPLES"] = self.options.examples
        cmake.definitions["CMAKE_BUILD_TYPE"] = self.settings.build_type

        return cmake

    def build(self):
        cmake = self._configure_cmake()
        cmake.configure(source_folder="qss-qasm")
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
        cmake = self._configure_cmake()
        cmake.install()
        self.copy("*.tab.cpp", dst="include/lib/Parser", src="lib/Parser")
        self.copy("*.tab.h", dst="include/lib/Parser", src="lib/Parser")

    def package_info(self):
        self.cpp_info.libs = ["qasmParser", "qasmFrontend", "qasmAST", "qasmDIAG"]
