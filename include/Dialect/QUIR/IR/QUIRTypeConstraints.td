//===- QUIRTypeConstraints.td - QUIR dialect type groups ---*- tablegen -*-===//
//
// (C) Copyright IBM 2022, 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#ifndef QUIR_TYPECONSTRAINTS
#define QUIR_TYPECONSTRAINTS

include "mlir/IR/OpBase.td"

def AnyQubit : Type<CPred<"$_self.isa<::mlir::quir::QubitType>()">, "Qubit">;

class Qubit<int width> : Type<CPred<[{$_self.isa<::mlir::quir::QubitType>() &&
        $_self.cast<::mlir::quir::QubitType>().getWidth() ==
        }] # width>, width # "-bit Qubit register"> {
                    int bitwidth = width;
                }

def AnyCBit : Type<Or<[
                        CPred<"$_self.isa<::mlir::quir::CBitType>()">,
                        CPred<"$_self.isInteger(1)">
                      ]>, "CBit">;

def AnyCBitOrSignlessInteger : Type<Or<[
                        CPred<"$_self.isa<::mlir::quir::CBitType>()">,
                        CPred<"$_self.isSignlessInteger()">
                      ]>, "CBitOrInteger">;

// Angle type - an unsigned fixed point number with variable precision, f64 representation
def AnyAngle : Type<CPred<"$_self.isa<::mlir::quir::AngleType>()">, "Angle">;

def AnyAngleOrQubit : AnyTypeOf<[AnyAngle, AnyQubit]>;

def AnyAngleOrCBit : AnyTypeOf<[AnyAngle, AnyCBit]>;

// Duration or stretch
def AnyDuration : Type<CPred<"$_self.isa<::mlir::quir::DurationType>()">, "Duration">;
def AnyStretch : Type<CPred<"$_self.isa<::mlir::quir::StretchType>()">, "Stretch">;
def DurationOrStretch : AnyTypeOf<[AnyDuration, AnyStretch]>;

def AnyClassical : AnyTypeOf<[AnyAngle, AnyCBit, DurationOrStretch, AnyInteger, AnyFloat, AnyComplex, Index]>;

#endif // QUIR_TYPECONSTRAINTS
