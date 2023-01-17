// RUN: qss-compiler -X=mlir --add-shot-loop %s | FileCheck %s

func @main() {
  sys.init
  // CHECK: scf.for
  // CHECK: quir.shot_init
  // CHECK: quir.shotLoop
  sys.finalize
  return
}
