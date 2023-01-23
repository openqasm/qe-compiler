# (C) Copyright IBM 2021.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

set(SOURCES LegacyInputConversion.cpp SystemDefinition.cpp DebugIndent.cpp)

find_package(boost 1.77.0 REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_library(Utils ${SOURCES})
target_link_libraries(Utils ${BOOST_LIBRARIES})
