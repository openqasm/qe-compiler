# (C) Copyright IBM 2023.
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

add_mlir_dialect_library(MLIRQUIRDialect
        QUIRDialect.cpp
        QUIRInterfaces.cpp
        QUIROps.cpp
        QUIRPatterns.cpp
        QUIRTestInterfaces.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/QUIR

        DEPENDS
        MLIRQUIRIncGen
        MLIRQUIRPatternsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
        MLIRQCSDialect
	)
