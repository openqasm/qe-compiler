# (C) Copyright IBM 2022.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

add_mlir_dialect_library(MLIRQUIRTransforms
	Analysis.cpp
    AddShotLoop.cpp
    AngleConversion.cpp
    BreakReset.cpp
    FunctionArgumentSpecialization.cpp
    LoadElimination.cpp
    MergeMeasures.cpp
    MergeParallelResets.cpp
    Passes.cpp
    QuantumDecoration.cpp
    RemoveQubitArgs.cpp
    SubroutineCloning.cpp
    UnusedVariable.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/QUIR

    DEPENDS
    MLIRQUIRIncGen
    MLIRQUIRPatternsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
