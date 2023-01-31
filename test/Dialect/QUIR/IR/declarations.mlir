// RUN: qss-compiler -X=mlir %s | FileCheck %s
//
// This test case checks that QUIR declarations can be parsed from
// textual/assembly input.
module {
    func @bar() {
        // CHECK: %{{.*}} = quir.declare_qubit : !quir.qubit<1>
        %qa1 = quir.declare_qubit : !quir.qubit<1>
        %qb1 = quir.declare_qubit : !quir.qubit<1>
        %qc1 = quir.declare_qubit : !quir.qubit<1>
        // CHECK: %{{.*}} = quir.declare_qubit : !quir.qubit<1>
        %qd1 = quir.declare_qubit : !quir.qubit<1>
        // CHECK: %{{.*}} = quir.constant #quir.angle<1.000000e-01 : !quir.angle<1>>
        %theta = quir.constant #quir.angle<0.1 : !quir.angle<1>>
        // CHECK: %{{.*}} = quir.constant #quir.angle<2.000000e-01  : !quir.angle>
        %mu = quir.constant #quir.angle<0.2 : !quir.angle>
        // CHECK %{{.*}} = oq3.declare_duration {value = "10ns"} : !quir.duration
        %len1 = "oq3.declare_duration"() {value = "10ns"} : () -> !quir.duration
        // CHECK %{{.*}} = oq3.declare_stretch : !quir.stretch
        %s1 = "oq3.declare_stretch"() : () -> !quir.stretch
        // CHECK %{{.*}} = oq3.declare_stretch : !quir.stretch
        %s2 = oq3.declare_stretch : !quir.stretch
        oq3.variable_decl { input } @flags : !quir.cbit<32>
        oq3.variable_decl { output } @result : !quir.cbit<1>
        return
    }
}
