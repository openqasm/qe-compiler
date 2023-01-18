// RUN: qss-compiler -X=mlir %s | FileCheck %s

module {
    func @bar() {
        %0 = arith.constant 1 : i32
        // CHECK: %{{.*}} = quir.declare_qubit : !quir.qubit<1>
        %qa1 = quir.declare_qubit : !quir.qubit<1>
        %qb1 = quir.declare_qubit : !quir.qubit<1>
        %qc1 = quir.declare_qubit : !quir.qubit<1>
        // CHECK: quir.reset %{{.*}} : !quir.qubit<1>
        quir.reset %qa1 : !quir.qubit<1>
        quir.reset %qb1 : !quir.qubit<1>
        quir.reset %qc1 : !quir.qubit<1>
        // CHECK: quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
        %res1 = "quir.measure"(%qb1) : (!quir.qubit<1>) -> i1
        // SYNCH: quir.measure(%{{.*}}) : (!quir.qubit<1>) -> i1
        // SYNCH-NEXT: sys.synchronize %{{.*}} : (!quir.qubit<1>) -> ()
        return
    }
}
