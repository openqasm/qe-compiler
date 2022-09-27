// RUN: qss-compiler -X=mlir %s | FileCheck %s

%c1 = arith.constant 0 : i1
%c2 = arith.constant 1 : i1
%o1 = arith.constant 10 : i32
%o2 = arith.constant 11 : i32
%o3 = arith.constant 13 : i32
%r1 = scf.if %c1 -> (i32) {
    %res1 = arith.addi %o1, %o2 : i32
    scf.yield %res1 : i32
} else {
    %res1 = arith.addi %o1, %o3 : i32
    scf.yield %res1 : i32
}
%out = arith.addi %r1, %o2 : i32

// qubit q;
// qubit r;
// angle[3] c = 0;
%qq1 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
%qr1 = quir.declare_qubit {id = 1 : i32} : !quir.qubit<1>
%angleC = quir.constant #quir.angle<0.0 : !quir.angle<3>>
// reset q;
// reset r;
quir.reset %qq1 : !quir.qubit<1>
quir.reset %qr1 : !quir.qubit<1>
// h r;
"quir.call_gate"(%qr1) {callee = @H} : (!quir.qubit<1>) -> ()
// uint[3] power = 1;
%power1 = arith.constant 1 : i3
// CHECK: [[CONST0:%.*]] = arith.constant 0 : index
%lb = arith.constant 0 : index
// CHECK: [[CONST2:%.*]] = arith.constant 2 : index
%ub = arith.constant 2 : index
// CHECK: [[CONST1:%.*]] = arith.constant 1 : index
%step = arith.constant 1 : index
// for i in [0: 2] {
// CHECK: %{{.*}}:2 = scf.for %{{.*}} = [[CONST0]] to [[CONST2]] step [[CONST1]] iter_args({{.*}}) -> (!quir.angle<3>, i3) {
%c_res, %p_res = scf.for %iv = %lb to %ub step %step
    iter_args(%angleC_iter = %angleC, %power1_iter = %power1) -> (!quir.angle<3>, i3) {
//  reset q;
    quir.reset %qq1 : !quir.qubit<1>
//   h q;
    "quir.call_gate"(%qq1) {callee = @H} : (!quir.qubit<1>) -> ()
//   cphase(power*3*pi/8) q, r;
    %power1_angle = "quir.cast"(%power1_iter) : (i3) -> !quir.angle<3>
    %angle_multiplicand = quir.constant #quir.angle<0.375 : !quir.angle<3>>
    %angleP = quir.angle_mul %power1_angle, %angle_multiplicand : !quir.angle<3>
    "quir.call_gate"(%qq1, %qr1, %angleP) {callee = @cphase} : (!quir.qubit<1>, !quir.qubit<1>, !quir.angle<3>) -> ()
    %angle_zero = quir.constant #quir.angle<0.0 : !quir.angle<3>>
    %negC = quir.angle_sub %angle_zero, %angleC_iter : !quir.angle<3>
//   phase(-c) q;
//   h q;
//   measure q -> c[0];
//   c <<= 1;
//   power <<= 1;
    %angleC_shift = quir.constant #quir.angle<0.0 : !quir.angle<3>>
    %power_shift = arith.constant 1 : i3
// }
// CHECK: scf.yield %{{.*}}, %{{.*}} : !quir.angle<3>, i3
    scf.yield %angleC_shift, %power_shift : !quir.angle<3>, i3
}
