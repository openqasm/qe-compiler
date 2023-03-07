//===- QUIRTypeConstraints.td - QUIR dialect type groups ---*- tablegen -*-===//
//
// (C) Copyright IBM 2022, 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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

def AnyCbit : Type<Or<[
                        CPred<"$_self.isa<::mlir::quir::CBitType>()">,
                        CPred<"$_self.isInteger(1)">
                      ]>, "CBit">;

def AnyCbitOrSignlessInteger : Type<Or<[
                        CPred<"$_self.isa<::mlir::quir::CBitType>()">,
                        CPred<"$_self.isSignlessInteger()">
                      ]>, "CBitOrInteger">;

// Angle type - an unsigned fixed point number with variable precision, f64 representation
def AnyAngle : Type<CPred<"$_self.isa<::mlir::quir::AngleType>()">, "Angle">;

def AnyAngleOrQubit : AnyTypeOf<[AnyAngle, AnyQubit]>;

def AnyAngleOrCbit : AnyTypeOf<[AnyAngle, AnyCbit]>;

// Duration or stretch
def AnyDuration : Type<CPred<"$_self.isa<::mlir::quir::DurationType>()">, "Duration">;
def AnyStretch : Type<CPred<"$_self.isa<::mlir::quir::StretchType>()">, "Stretch">;
def DurationOrStretch : AnyTypeOf<[AnyDuration, AnyStretch]>;

def AnyClassical : AnyTypeOf<[AnyAngle, AnyCbit, DurationOrStretch, AnyInteger, AnyFloat, AnyComplex, Index]>;

#endif // QUIR_TYPECONSTRAINTS
