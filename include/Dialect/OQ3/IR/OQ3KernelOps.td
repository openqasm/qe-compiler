//===- OQ3KernelOps.td - OpenQASM 3 dialect kernel ops -----*- tablegen -*-===//
//
// (C) Copyright IBM 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
/// This is the operation definition file for OpenQASM 3 kernel operations.
///
//===----------------------------------------------------------------------===//

#ifndef OQ3_KERNEL_OPS
#define OQ3_KERNEL_OPS

include "mlir/Interfaces/CallInterfaces.td"

def OQ3_KernelCallOp : OQ3_Op<"kernel_call", [DeclareOpInterfaceMethods<CallOpInterface>]> {
    let summary = "A call to a user-defined kernel function that needs to be specialized for input operands";
    let description = [{
        The `oq3.kernel_call` operation represents calls to user-defined kernel function
        that needs to be specialized for its arguments. The callee function is attached to a symbol reference via
        an attribute. The arguments list must match the arguments provided by the callee. Kernels may only receive
        and return classical types.

        Example:
        ```mlir
        %majority = oq3.kernel_call @majority_vote(%cbitarray) : (memref<3xi1>) -> i1
        ```
        This is valid only if the named kernel @majority_vote exists and takes a cbit array as arguments and
        returns an i1 as a result.
    }];

    let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyClassical>:$operands);
    let results = (outs Optional<AnyClassical>:$res);

    let assemblyFormat = [{
        $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
    }];

    let builders = [
        OpBuilder<(ins "FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", SymbolRefAttr::get(callee));
            $_state.addTypes(callee.getType().getResults());
        }]>,
        OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$res,
        CArg<"ValueRange", "{}">:$operands), [{
            $_state.addOperands(operands);
            $_state.addAttribute("callee", callee);
            $_state.addTypes(res);
        }]>,
        OpBuilder<(ins "StringRef":$callee, "TypeRange":$res,
        CArg<"ValueRange", "{}">:$operands), [{
            build($_builder, $_state, SymbolRefAttr::get($_builder.getContext(), callee), res,
                    operands);
        }]>];
}

#endif // OQ3_KERNEL_OPS
