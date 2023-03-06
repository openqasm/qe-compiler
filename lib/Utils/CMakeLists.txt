# (C) Copyright IBM 2021, 2023.
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

set(SOURCES LegacyInputConversion.cpp SystemDefinition.cpp DebugIndent.cpp)

include_directories(${Boost_INCLUDE_DIRS})

add_library(Utils ${SOURCES})
target_link_libraries(Utils ${BOOST_LIBRARIES})
