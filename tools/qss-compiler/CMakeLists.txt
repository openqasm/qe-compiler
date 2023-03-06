# (C) Copyright IBM 2021, 2023.
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

add_llvm_executable(qss-compiler qss-compiler.cpp)
llvm_update_compile_flags(qss-compiler)
target_link_libraries(qss-compiler PRIVATE QSSCLib)
mlir_check_all_link_libraries(qss-compiler)
