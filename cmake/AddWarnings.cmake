# (C) Copyright IBM 2024.
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

# Based on LLVMs:
# llvm-project/llvm/cmake/modules/HandleLLVMOptions.cmake

# uses the same set of options as LLVM_ENABLE_WARNINGS except for:
# covered-switch-default has been disabled

include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)

option(QSSC_ENABLE_WARNINGS "Enable LLVM style warnings" ON)
message(STATUS "QSS Warnings Enabled: ${QSSC_ENABLE_WARNINGS}")

if (QSSC_ENABLE_WARNINGS)

  function(append value)
    foreach(variable ${ARGN})
      set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
    endforeach(variable)
  endfunction()

  macro(add_flag_if_supported flag name)
    check_c_compiler_flag("-Werror ${flag}" "C_SUPPORTS_${name}")
    append_if("C_SUPPORTS_${name}" "${flag}" CMAKE_C_FLAGS)
    check_cxx_compiler_flag("-Werror ${flag}" "CXX_SUPPORTS_${name}")
    append_if("CXX_SUPPORTS_${name}" "${flag}" CMAKE_CXX_FLAGS)
  endmacro()

  function(append_if condition value)
    if (${condition})
        foreach(variable ${ARGN})
        set(${variable} "${${variable}} ${value}" PARENT_SCOPE)
        endforeach(variable)
    endif()
  endfunction()

  append("-Wall" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)

  append("-Wextra -Wno-unused-parameter -Wwrite-strings" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  append("-Wcast-qual" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)

  # Turn off missing field initializer warnings for gcc to avoid noise from
  # false positives with empty {}. Turn them on otherwise (they're off by
  # default for clang).
  check_cxx_compiler_flag("-Wmissing-field-initializers" CXX_SUPPORTS_MISSING_FIELD_INITIALIZERS_FLAG)
  if (CXX_SUPPORTS_MISSING_FIELD_INITIALIZERS_FLAG)
    if (CMAKE_COMPILER_IS_GNUCXX)
      append("-Wno-missing-field-initializers" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    else()
      append("-Wmissing-field-initializers" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
    endif()
  endif()

  add_flag_if_supported("-Wimplicit-fallthrough" IMPLICIT_FALLTHROUGH_FLAG)
  add_flag_if_supported("-Wno-covered-switch-default" COVERED_SWITCH_DEFAULT_FLAG)
  append_if(USE_NO_UNINITIALIZED "-Wno-uninitialized" CMAKE_CXX_FLAGS)
  append_if(USE_NO_MAYBE_UNINITIALIZED "-Wno-maybe-uninitialized" CMAKE_CXX_FLAGS)

  # Disable -Wnonnull for GCC warning as it is emitting a lot of false positives.
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    append("-Wno-nonnull" CMAKE_CXX_FLAGS)
  endif()

  # Disable -Wredundant-move and -Wpessimizing-move on GCC>=9. GCC wants to
  # remove std::move in code like "A foo(ConvertibleToA a) {
  # return std::move(a); }", but this code does not compile (or uses the copy
  # constructor instead) on clang<=3.8. Clang also has a -Wredundant-move and
  # -Wpessimizing-move, but they only fire when the types match exactly, so we
  # can keep them here.
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    check_cxx_compiler_flag("-Wredundant-move" CXX_SUPPORTS_REDUNDANT_MOVE_FLAG)
    append_if(CXX_SUPPORTS_REDUNDANT_MOVE_FLAG "-Wno-redundant-move" CMAKE_CXX_FLAGS)
    check_cxx_compiler_flag("-Wpessimizing-move" CXX_SUPPORTS_PESSIMIZING_MOVE_FLAG)
    append_if(CXX_SUPPORTS_PESSIMIZING_MOVE_FLAG "-Wno-pessimizing-move" CMAKE_CXX_FLAGS)
  endif()

  # The LLVM libraries have no stable C++ API, so -Wnoexcept-type is not useful.
  check_cxx_compiler_flag("-Wnoexcept-type" CXX_SUPPORTS_NOEXCEPT_TYPE_FLAG)
  append_if(CXX_SUPPORTS_NOEXCEPT_TYPE_FLAG "-Wno-noexcept-type" CMAKE_CXX_FLAGS)

  # Check if -Wnon-virtual-dtor warns for a class marked final, when it has a
  # friend declaration. If it does, don't add -Wnon-virtual-dtor. The case is
  # considered unhelpful (https://gcc.gnu.org/PR102168).
  set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -Werror=non-virtual-dtor")
  CHECK_CXX_SOURCE_COMPILES("class f {};
                             class base {friend f; public: virtual void anchor();protected: ~base();};
                             int main() { return 0; }"
                            CXX_WONT_WARN_ON_FINAL_NONVIRTUALDTOR)
  set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
  append_if(CXX_WONT_WARN_ON_FINAL_NONVIRTUALDTOR "-Wnon-virtual-dtor" CMAKE_CXX_FLAGS)

  append("-Wdelete-non-virtual-dtor" CMAKE_CXX_FLAGS)

  # Enable -Wsuggest-override if it's available, and only if it doesn't
  # suggest adding 'override' to functions that are already marked 'final'
  # (which means it is disabled for GCC < 9.2).
  check_cxx_compiler_flag("-Wsuggest-override" CXX_SUPPORTS_SUGGEST_OVERRIDE_FLAG)
  if (CXX_SUPPORTS_SUGGEST_OVERRIDE_FLAG)
    set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
    set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -Werror=suggest-override")
    CHECK_CXX_SOURCE_COMPILES("class base {public: virtual void anchor();};
                               class derived : base {public: void anchor() final;};
                               int main() { return 0; }"
                              CXX_WSUGGEST_OVERRIDE_ALLOWS_ONLY_FINAL)
    set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
    append_if(CXX_WSUGGEST_OVERRIDE_ALLOWS_ONLY_FINAL "-Wsuggest-override" CMAKE_CXX_FLAGS)
  endif()

  # Check if -Wcomment is OK with an // comment ending with '\' if the next
  # line is also a // comment.
  set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -Werror -Wcomment")
  CHECK_C_SOURCE_COMPILES("// \\\\\\n//\\nint main(void) {return 0;}"
                          C_WCOMMENT_ALLOWS_LINE_WRAP)
  set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})
  if (NOT C_WCOMMENT_ALLOWS_LINE_WRAP)
    append("-Wno-comment" CMAKE_C_FLAGS CMAKE_CXX_FLAGS)
  endif()

  # Enable -Wstring-conversion to catch misuse of string literals.
  add_flag_if_supported("-Wstring-conversion" STRING_CONVERSION_FLAG)

  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Disable the misleading indentation warning with GCC; GCC can
    # produce noisy notes about this getting disabled in large files.
    # See e.g. https://gcc.gnu.org/bugzilla/show_bug.cgi?id=89549
    check_cxx_compiler_flag("-Wmisleading-indentation" CXX_SUPPORTS_MISLEADING_INDENTATION_FLAG)
    append_if(CXX_SUPPORTS_MISLEADING_INDENTATION_FLAG "-Wno-misleading-indentation" CMAKE_CXX_FLAGS)
  else()
    # Prevent bugs that can happen with llvm's brace style.
    add_flag_if_supported("-Wmisleading-indentation" MISLEADING_INDENTATION_FLAG)
  endif()

  # Enable -Wctad-maybe-unsupported to catch unintended use of CTAD.
  add_flag_if_supported("-Wctad-maybe-unsupported" CTAD_MAYBE_UNSPPORTED_FLAG)
endif (QSSC_ENABLE_WARNINGS)
