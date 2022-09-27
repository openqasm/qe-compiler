//===- QUIRAttributes.td - QUIR dialect attributes ---------*- tablegen -*-===//
//
// (C) Copyright IBM 2022.
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

#endif // QUIR_ATTRIBUTES
