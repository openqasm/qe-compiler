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

include "mlir/IR/OpBase.td"

def QUIR_AngleAttr : QUIR_Attr<"Angle", "angle"> {
  let summary = "A floating point value in the range (0,2*pi]";
  let description = [{
    Angle attributes represent constant known floating point values within the
    range (0,2*pi]
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "APFloat":$value);

  let builders = [
    AttrBuilderWithInferredContext<(ins "Type":$type,
                                        "const APFloat &":$value), [{
      return $_get(type.getContext(), type, value);
    }]>,
    AttrBuilder<(ins "Type":$type, "const APFloat &":$value), [{
      return $_get(context, type, value);
    }]>
  ];

  let assemblyFormat = "`<` $value `:` $type `>`";
  let skipDefaultBuilders = 1;
}

def QUIR_DurationAttr : QUIR_Attr<"Duration", "duration"> {
  let summary = "A duration of time";
  let description = [{
    Duration attributes represent OpenQASM 3.0 durations, which are lengths of
    time. They are represented by a string containing a float literal and unit
    specifiers. Valid units are s, ms, ns, us, and dt.
  }];

  let parameters = (ins AttributeSelfTypeParameter<"">:$type,
                        StringRefParameter<"the duration">:$durationString);

  let assemblyFormat = "`<` $durationString `:` $type `>`";

  let extraClassDeclaration = [{
        Duration getDuration();
        double getValue();
        Duration::DurationUnit getUnits();
    }];
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
