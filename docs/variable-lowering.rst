OpenQASM 3 Variable Lowering
============================

The current lowering pipeline for variables is driven by the goals of
enabling efficient optimizations based on the static single assignment
(SSA) form's convenient use-def chains while at the same time keeping
the difference of abstraction bridged in each transformation step low.

As an example, consider the OpenQASM 3 code

::

   a = measure $0;
   if (a)
     x $0;

To enable optimizations to have a clear “view” of data flow between
classical and quantum operations, we aim to promote all variable
operations into SSA form such as

::

   %0 = quir.declare_qubit {id = 0 : i32}
   %1 = quir.measure(%0)
   scf.if %1 {
     quir.call_gate @x(%0)
   }

While that connection between the measurement and the x-gate’s condition
is obvious from the OpenQASM 3 code for a human reader, it takes a
compiler several transformation steps to reach the MLIR example code
above.

The qss-compiler proceeds as follows: Statements involving OpenQASM 3
variables are lowered to MLIR’s built-in ``memref`` dialect in several
steps. The initial MLIR follows the memory semantics of OpenQASM 3. Two
initial optimizations can remove some variables. Finally, variable
declarations, assignments, and references are converted to memory
operations from MLIR’s builtin dialects and replaced with SSA values in
many cases (not a full SSA transformation, though).

QUIRGenQASM3Visitor
-------------------

The OpenQASM 3 frontend’s first step that generates MLIR, the
`QUIRGenQASM3Visitor <https://github.com/Qiskit/qss-compiler/blob/main/lib/Frontend/OpenQASM3/QUIRGenQASM3Visitor.cpp>`__ ,
follows the memory semantics of OpenQASM 3 variables: each variable
identifies a location in memory and each reference or assignment to that
variables reads or writes that location.

-  Variables are declared by the QUIR operation
   ``oq3.declare_variable`` and identified as `MLIR
   symbols <https://mlir.llvm.org/docs/SymbolsAndSymbolTables/>`__.
-  Each reference to a variable is modeled as an operation
   ``quir.use_variable`` that returns the variables value at that point.
-  The operation ``oq3.assign_variable`` updates the variable’s value
   (visible from that operation forward until a subsequent
   ``oq3.assign_variable`` that operates on the same variable).
-  Both ``quir.use_variable`` and ``oq3.assign_variable`` refer the
   MLIR symbol defined by the operations ``oq3.declare_variable`` (the
   symbol’s name is a string).

As an example, the OpenQASM 3 statement ``a = a ^ b`` (with ``a`` and
``b`` both declared as ``bit``, will yield MLIR operations (simplified
for clarity):

::

      %0 = quir.use_variable @a
      %1 = quir.use_variable @b
      %2 = quir.cbit_xor %s02, %s13
      oq3.assign_variable @a = %2

Variable scoping is not supported yet (there is only the global scope). Adding
support is future work.

Note that MLIR is always in SSA form (by construction). Yet, the SSA
use-def chains between the operations that actually define and use
variable values are broken by `assign_variable` and `use_variable`
operations.


Coarse Initial Optimizations
----------------------------

Two initial optimization steps can remove (or simplify) variables
altogether (saving memory and potentially enabling further optimizations
of values assigned to them). The
`LoadEliminationPass <https://github.com/Qiskit/qss-compiler/blob/main/lib/Dialect/QUIR/Transforms/LoadElimination.cpp>`__
can remove variables that are only assigned once and, thus, effectively
act as constants (replacing every use with the single assigned value).
The
`UnusedVariablePass <https://github.com/Qiskit/qss-compiler/blob/main/lib/Dialect/QUIR/Transforms/UnusedVariable.cpp>`__
removes variables that are never referenced.

Best-Effort SSA Transformation
------------------------------

The
`VariableEliminationPass <https://github.com/Qiskit/qss-compiler/blob/main/lib/Dialect/QUIR/Transforms/VariableElimination.cpp>`__
reuses the scalar-replacement pass from MLIR’s affine dialect to
replace instances of `quir.use_variable` with the MLIR Value previously
assigned to the respective variable, in many cases. Noteably, that code
does not perform a complete SSA transformation and variable assignment
and use around control flow will be left as memory operations. The
(partial) transformation has four steps:

Lowering to Memory Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The operations ``oq3.declare_variable``, ``quir.use_variable``, and
``oq3.assign_variable`` are converted to MLIR’s ``memref`` and
``affine`` dialects. Each variable declaration is turned into a global
variable (``memref.global``) and variable reads/writes are converted
into ``load`` and ``store`` operations from the ``affine`` dialect (only
for the purpose of reusing that dialect’s scalar replacement pass).

Prefer Stack Allocation
~~~~~~~~~~~~~~~~~~~~~~~

To accomodate the affine dialect’s scalar replacement pass’s relatively
simple alias analysis, global variables are replaced by stack allocated
variables whenever possible (i.e., when they are not externally
visible).

Affine Scalar Replacement
~~~~~~~~~~~~~~~~~~~~~~~~~

Employ the affine dialect’s scalar replacement pass (the implementation,
not the pass scaffolding) to replace (some) memory loads with the
forwarded values from previous stores (store to load forwarding).

See also the `MLIR documentation on
-affine-scalrep <https://mlir.llvm.org/docs/Passes/#-affine-scalrep-replace-affine-memref-accesses-by-scalars-by-forwarding-stores-to-loads-and-eliminating-redundant-loads>`__.

Eliminate Isolated Stores
~~~~~~~~~~~~~~~~~~~~~~~~~

Finally, a pattern removes all remaining stack-local variables that are
only written yet never read. This optimization is sound since the
variables are not visible outside the program (only on its runtime
stack) and their contents are not used in the program (there are no
reads, and there are no aliases that could access them).
