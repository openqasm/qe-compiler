// RUN: qss-compiler -X=mlir --add-shot-loop %s | FileCheck %s

func @main() {
  sys.init
  // CHECK: scf.for
  // CHECK: sys.shot_loop_init
  // CHECK: sys.shot_loop
  sys.finalize
  return
}
