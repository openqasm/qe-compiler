// RUN: qss-compiler -X=mlir --add-shot-loop %s | FileCheck %s

func @main() {
  quir.system_init
  // CHECK: scf.for
  // CHECK: quir.shot_init
  // CHECK: quir.shotLoop
  quir.system_finalize
  return
}
