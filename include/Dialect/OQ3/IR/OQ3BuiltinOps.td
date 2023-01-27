//===- OQ3BuiltinOps.td - OpenQASM 3 dialect builtin ops ---*- tablegen -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This is the operation definition file for OpenQASM 3 builtin operations.
///
//===----------------------------------------------------------------------===//

#ifndef OQ3_BUILTIN_OPS
#define OQ3_BUILTIN_OPS

// Define a side effect that identifies an operation as not dead while not
// interfering with memory operations (e.g., allows store-forwarding across
// this operation).
// Note that operations without memory effects defined will be treated
// conservatively (i.e., not making any assumptions).
// see lib/Interfaces/SideEffectInterfaces.cpp:isTriviallyDeadImpl()
// see lib/Dialect/Affine/Utils/Utils.cpp:hasNoInterveningEffect()
// def NonInterferingNonDeadSideEffect : MemoryEffects<[MemFree<DefaultResource>]>;

// def OQ3_BuiltinCXOp : OQ3_Op<"builtin_CX", [
//             UnitaryOp,
//             DeclareOpInterfaceMethods<QubitOpInterface, ["getOperatedQubits"]>,
//             NonInterferingNonDeadSideEffect]> {
//     let summary = "Controlled NOT gate";
//     let description = [{
//         The `quir.builtin_CX` operation performs a controlled not i.e. flips %target iff %control is a 1.
//         Example:
//         ```mlir
//         quir.builtin_CX %control, %target : !quir.qubit<1>, !quir.qubit<1>
//         ```
//     }];

//     let arguments = (ins AnyQubit:$control, AnyQubit:$target);

//     let assemblyFormat = [{
//         attr-dict $control `,` $target `:` type($control) `,` type($target)
//     }];
// }

// def OQ3_BuiltinUOp : OQ3_Op<"builtin_U", [
//             UnitaryOp,
//             DeclareOpInterfaceMethods<QubitOpInterface, ["getOperatedQubits"]>,
//             NonInterferingNonDeadSideEffect]> {
//     let summary = "Generic unitary gate";
//     let description = [{
//         The `quir.builtin_U` operation performs a single qubit unitary rotation.
//         Example:
//         ```mlir
//         quir.builtin_U %target, %theta_0, %phi_0, %lambda_0 : !quir.qubit<1>, !quir.angle<1>, !quir.angle<1>, !quir.angle<1>
//         ```
//     }];

//     let arguments = (ins AnyQubit:$target, AnyAngle:$theta, AnyAngle:$phi, AnyAngle:$lambda);

//     let assemblyFormat = [{
//         attr-dict $target `,` $theta `,` $phi `,` $lambda `:` type($target) `,` type($theta) `,` type($phi) `,` type($lambda)
//     }];
// }

#endif // OQ3_BUILTIN_OPS
