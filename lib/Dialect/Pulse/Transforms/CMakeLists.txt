# (C) Copyright IBM 2022.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

add_mlir_dialect_library(MLIRPulseTransforms
        InlineRegion.cpp
        Scheduling.cpp
        SlicePorts.cpp
        SystemCreation.cpp
        Passes.cpp
        PortGroupPrune.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Pulse

        DEPENDS
        MLIRPulseIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	Utils
	)
