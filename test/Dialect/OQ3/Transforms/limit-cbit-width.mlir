// RUN: qss-compiler -X=mlir -emit=mlir %s --oq3-limit-cbit-width | FileCheck %s
//
// Test that the we can successfully split cbit arrays larger than 32 bits wide

module {
  
  %c34359738368_i36= arith.constant 34359738368 : i36
  %c0_i2 = arith.constant 0 : i2
  %c0_i48 = arith.constant 0 : i48
  %c0_i4 = arith.constant 0 : i4

  // test does not change cbit arrays < 32
  oq3.declare_variable @meas : !quir.cbit<4>
  // CHECK: oq3.declare_variable @meas : !quir.cbit<4>

  // test breaking apart wide array
  oq3.declare_variable @wide : !quir.cbit<48>
  // CHECK-NOT: oq3.declare_variable @wide : !quir.cbit<48>
  // CHECK: oq3.declare_variable @wide_00 : !quir.cbit<32>
  // CHECK: oq3.declare_variable @wide_1 : !quir.cbit<16>
  
  // test does not overwrite existing symbols when breaking appart wide arrays
  oq3.declare_variable @wide_0 : !quir.cbit<2>
  // CHECK: oq3.declare_variable @wide_0 : !quir.cbit<2>

  // test wide array assignment 
  oq3.declare_variable @assignment1 : !quir.cbit<36>
  // CHECK-NOT: oq3.declare_variable @assignment1 : !quir.cbit<36>
  // CHECK: oq3.declare_variable @assignment1_0 : !quir.cbit<32>
  // CHECK: oq3.declare_variable @assignment1_1 : !quir.cbit<4>
  
  // test initialization into original narrow array - no change
  %0 = "oq3.cast"(%c0_i4) : (i4) -> !quir.cbit<4>
  oq3.variable_assign @meas : !quir.cbit<4> = %0
  // CHECK: %0 = "oq3.cast"(%c0_i4) : (i4) -> !quir.cbit<4>
  // CHECK: oq3.variable_assign @meas : !quir.cbit<4> = %0

  // test initialization of split array
  %1 = "oq3.cast"(%c0_i48) : (i48) -> !quir.cbit<48>
  oq3.variable_assign @wide : !quir.cbit<48> = %1
  // CHECK: [[CAST1:%.*]] = "oq3.cast"(%c0_i32) : (i32) -> !quir.cbit<32>
  // CHECK: oq3.variable_assign @wide_00 : !quir.cbit<32> = [[CAST1]]
  // CHECK: %c0_i16 = arith.constant 0 : i16
  // CHECK: [[CAST2:%.*]]  = "oq3.cast"(%c0_i16) : (i16) -> !quir.cbit<16>
  // CHECK: oq3.variable_assign @wide_1 : !quir.cbit<16> = [[CAST2]]
  

  // test initialization into original narrow array symbol name that 
  // would have been used to break apart wide wide array - no change
  %2 = "oq3.cast"(%c0_i2) : (i2) -> !quir.cbit<2>
  oq3.variable_assign @wide_0 : !quir.cbit<2> = %2
  // CHECK: [[INIT1:%.*]] = "oq3.cast"(%c0_i2) : (i2) -> !quir.cbit<2>
  // CHECK: oq3.variable_assign @wide_0 : !quir.cbit<2> = [[INIT1]]
  
  // declare qubits to make sure measurement mapping is correct
  %3 = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  %4 = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
  %5 = quir.declare_qubit {id = 47 : i32} : !quir.qubit<1>
  // CHECK: [[Q0:%.*]] = quir.declare_qubit {id = 0 : i32} : !quir.qubit<1>
  // CHECK: [[Q3:%.*]] = quir.declare_qubit {id = 3 : i32} : !quir.qubit<1>
  // CHECK: [[Q47:%.*]] = quir.declare_qubit {id = 47 : i32} : !quir.qubit<1>
  
  // assignment into narrow array - no change
  %7 = quir.measure(%4) : (!quir.qubit<1>) -> i1
  oq3.cbit_assign_bit @meas<4> [3] : i1 = %7
  // CHECK: [[M1:%.*]] = quir.measure([[Q3]]) : (!quir.qubit<1>) -> i1
  // CHECK: oq3.cbit_assign_bit @meas<4> [3] : i1 = [[M1]]
  
  // assignment into wide array LSB - assign into lowest split register
  %8 = quir.measure(%3) : (!quir.qubit<1>) -> i1
  oq3.cbit_assign_bit @wide<48> [0] : i1 = %8
  // CHECK: [[M2:%.*]] = quir.measure([[Q0]]) : (!quir.qubit<1>) -> i1
  // CHECK: oq3.cbit_assign_bit @wide_00<32> [0] : i1 = [[M2]]

  // assignment into wide array MSB - 
  // assign into calculated  split register
  %10 = quir.measure(%5) : (!quir.qubit<1>) -> i1
  oq3.cbit_assign_bit @wide<48> [47] : i1 = %10
  // CHECK: [[M4:%.*]] = quir.measure([[Q47]]) : (!quir.qubit<1>) -> i1
  // CHECK: oq3.cbit_assign_bit @wide_1<16> [15] : i1 = [[M4]]

  // test intialization of wide array 
  %11 = "oq3.cast"(%c34359738368_i36) : (i36) -> !quir.cbit<36>
  oq3.variable_assign @assignment1 : !quir.cbit<36> = %11
  %12 = oq3.variable_load @assignment1 : !quir.cbit<36>
  // CHECK: [[CAST1:%.*]] = "oq3.cast"(%c0_i32_0) : (i32) -> !quir.cbit<32>
  // CHECK: oq3.variable_assign @assignment1_0 : !quir.cbit<32> = [[CAST1:%.*]]
  // CHECK: [[CAST2:%.*]] = "oq3.cast"(%c-8_i4) : (i4) -> !quir.cbit<4>
  // CHECK: oq3.variable_assign @assignment1_1 : !quir.cbit<4> = [[CAST2:%.*]]

  
  // test remapped extract bit and if conditional
  %13 = oq3.cbit_extractbit(%12 : !quir.cbit<36>) [35] : i1
  scf.if %13 {
    %14 = oq3.variable_load @assignment1 : !quir.cbit<36>
    %15 = oq3.cbit_extractbit(%14 : !quir.cbit<36>) [0] : i1
    oq3.cbit_assign_bit @assignment1<36> [35] : i1 = %15
  }
  // CHECK: {{%.*}} = oq3.variable_load @assignment1_0 : !quir.cbit<32>
  // CHECK: {{%.*}} = oq3.variable_load @assignment1_1 : !quir.cbit<4>
  // CHECK: [[COND:%.*]] = oq3.cbit_extractbit(%15 : !quir.cbit<4>) [3] : i1
  // CHECK: scf.if [[COND]] {
  // CHECK: [[LOADLOWER:%.*]] = oq3.variable_load @assignment1_0 : !quir.cbit<32>
  // CHECK: [[LOADUPPER:%.*]] = oq3.variable_load @assignment1_1 : !quir.cbit<4>
  // CHECK: [[EXTRACTLOWER:%.*]] = oq3.cbit_extractbit([[LOADLOWER]] : !quir.cbit<32>) [0] : i1
  // CHECK: oq3.cbit_assign_bit @assignment1_1<4> [3] : i1 = [[EXTRACTLOWER]]
  
} 
