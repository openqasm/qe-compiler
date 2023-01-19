// RUN: qss-compiler -X=mlir --add-shot-loop %s | FileCheck %s

func @main() {
  qusys.init
  // CHECK: scf.for
  // CHECK: qusys.shot_init
  // CHECK: qusys.shot_loop
  qusys.finalize
  return
}
