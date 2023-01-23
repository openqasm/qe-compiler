// RUN: qss-compiler -X=mlir --add-shot-loop %s | FileCheck %s

func @main() {
  qcs.init
  // CHECK: scf.for
  // CHECK: qcs.shot_init
  // CHECK: qcs.shot_loop
  qcs.finalize
  return
}
