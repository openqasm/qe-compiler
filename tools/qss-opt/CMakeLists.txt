# (C) Copyright IBM 2021.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

add_llvm_executable(qss-opt qss-opt.cpp)
llvm_update_compile_flags(qss-opt)
target_link_libraries(qss-opt PRIVATE QSSCLib)

mlir_check_all_link_libraries(qss-opt)
