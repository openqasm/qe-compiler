// RUN: qss-opt %s | qss-opt | FileCheck %s
// Verify the printed output can be parsed.
// RUN: qss-opt %s --mlir-print-op-generic | qss-opt | FileCheck %s

func @extract(%in: !quir.cbit<2>) -> i1 {
  // CHECK: oq3.cbit_extractbit(%arg0 : !quir.cbit<2>) [1] : i1
  %1 = oq3.cbit_extractbit(%in : !quir.cbit<2>) [1] : i1
  // CHECK: return %0 : i1
  return %1 : i1
}

func @insert(%cbit: !quir.cbit<2>, %bit :i1) -> !quir.cbit<2> {
  // CHECK: oq3.cbit_insertbit(%arg0 : !quir.cbit<2>) [0] = %arg1 : !quir.cbit<2>
  %1 = oq3.cbit_insertbit(%cbit : !quir.cbit<2>)[0] = %bit : !quir.cbit<2>
  // CHECK: return %0 : !quir.cbit<2>
  return %1 : !quir.cbit<2>
}
