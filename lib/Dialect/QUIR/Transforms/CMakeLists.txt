# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0 with LLVM
# Exceptions. You may obtain a copy of this license in the LICENSE.txt
# file in the root directory of this source tree.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.


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
    RemoveQubitOperands.cpp
    ReorderMeasurements.cpp
    SubroutineCloning.cpp
    UnusedVariable.cpp
    VariableElimination.cpp
    AffineScalRepCopy.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/QUIR

    DEPENDS
    MLIRQUIRIncGen
    MLIRQUIRPatternsIncGen

	LINK_LIBS PUBLIC
	MLIRIR
	)
