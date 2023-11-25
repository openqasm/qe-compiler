//===- QUIRAttributes.td - QUIR dialect attributes ---------*- tablegen -*-===//
//
// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//


#ifndef QUIR_ATTRIBUTES
#define QUIR_ATTRIBUTES

include "Dialect/QUIR/IR/QUIRTypes.td"

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

def QUIR_AngleAttr : QUIR_Attr<"Angle", "angle", [TypedAttrInterface]> {
  let summary = "A floating point value in the range (0,2*pi]";
  let description = [{
    Angle attributes represent constant known floating point values within the
    range (0,2*pi]
  }];

  let parameters = (ins AttributeSelfTypeParameter<"", "AngleType">:$type, "APFloat":$value);

  let builders = [
    AttrBuilderWithInferredContext<(ins "AngleType":$type,
                                        "const APFloat &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>,
    AttrBuilder<(ins "AngleType":$type, "const APFloat &":$value), [{
      return $_get(context, type, value);
    }]>
  ];

  let assemblyFormat = "`<` $value `>`";
  let skipDefaultBuilders = 1;
}

def QUIR_DurationAttr : QUIR_Attr<"Duration", "duration", [TypedAttrInterface]> {
  let summary = "A duration of time";
  let description = [{
    Duration attributes represent a duration of time.
    These are represented by a floating point value and a duration
    units type.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"", "DurationType">:$type,
                        "APFloat":$duration);

  let extraClassDeclaration = [{

        /// Get the scheduling timestep length (in seconds) from the input scheduling rate (in seconds).
        static double getDtFromSchedulingRate(const double schedulingRate);
        /// Get the scheduling rate (in seconds) from the input scheduling timestep length (in seconds).
        static double getSchedulingRateFromDt(const double dt);

        /// Convert duration of inputUnits to outputUnits
        static double convertUnitsToUnits(double value, TimeUnits inputUnits, TimeUnits outputUnits, const double dt);

        /// Convert the duration attribute to an integer number
        /// of scheduling samples based on the input scheduling
        /// timestep length, dt.
        uint64_t getSchedulingCycles(const double dt);
        /// Convert this duration attribute's value to the target units.
        double convertUnits(const TimeUnits targetUnits, const double dt);
        /// Convert this duration attribute to a new duration attribute
        /// with the target units.
        DurationAttr getConvertedDurationAttr(const TimeUnits targetUnits, const double dt);
    }];

    let builders = [
    AttrBuilderWithInferredContext<(ins "DurationType":$type,
                                        "const APFloat &":$duration), [{
      return $_get(type.getContext(), type, duration);
    }]>,
    AttrBuilder<(ins "DurationType":$type, "const APFloat &":$duration), [{
      return $_get(context, type, duration);
    }]>
  ];

  let assemblyFormat = "`<` $duration `>`";
  let skipDefaultBuilders = 1;

}

def QUIR_InputParameterAttr : QUIR_Attr<"InputParameter", "inputParameter"> {
  let summary = "A marker for an operation that represents an input parameter";
  let description = [{
    InputParameter attributes mark operations that represent input parameters
    to a circuit.
  }];

  let parameters = (ins StringRefParameter<"the parameter's name">:$name);

  let assemblyFormat = "`<` $name `>`";
  let genAccessors = 1;
}

#endif // QUIR_ATTRIBUTES
