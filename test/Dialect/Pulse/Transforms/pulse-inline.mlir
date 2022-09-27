// RUN: qss-compiler -X=mlir --pulse-inline %s | FileCheck %s

// CHECK-NOT:   func @x0(%arg0: !pulse.port_group)
// CHECK-NOT:   func @x0(%arg0: !pulse.port_group)
// CHECK-NOT:   func @x0(%arg0: !pulse.port_group)
// CHECK-NOT:   func @x0(%arg0: !pulse.port_group)
// CHECK-NOT:   func @cx01(%arg0: !pulse.port_group, %arg1: !pulse.port_group)
// CHECK-NOT:   func @cx10(%arg0: !pulse.port_group, %arg1: !pulse.port_group)
// CHECK-NOT:   func @cx12(%arg0: !pulse.port_group, %arg1: !pulse.port_group)
// CHECK-NOT:   func @rz0(%arg0: !pulse.port_group, %arg1: !quir.angle<10>)
// CHECK-NOT:   func @rz1(%arg0: !pulse.port_group, %arg1: !quir.angle<10>)
// CHECK-NOT:   func @sx1(%arg0: !pulse.port_group)
// CHECK-NOT:   func @sx2(%arg0: !pulse.port_group)
// CHECK-NOT:   func @sx3(%arg0: !pulse.port_group)

// CHECK "pulse.select_port"(%arg0) {id = "d0"} : (!pulse.port_group) -> !pulse.port
// CHECK "pulse.select_port"(%arg0) {id = "d0"} : (!pulse.port_group) -> !pulse.port
// CHECK "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
// CHECK "pulse.select_port"(%arg0) {id = "d2"} : (!pulse.port_group) -> !pulse.port
// CHECK "pulse.select_port"(%arg0) {id = "d3"} : (!pulse.port_group) -> !pulse.port
// CHECK "pulse.select_port"(%arg0) {id = "u0"} : (!pulse.port_group) -> !pulse.port
// CHECK "pulse.select_port"(%arg0) {id = "u1"} : (!pulse.port_group) -> !pulse.port
// CHECK "pulse.select_port"(%arg0) {id = "u4"} : (!pulse.port_group) -> !pulse.port
// CHECK "pulse.select_port"(%arg0) {id = "u8"} : (!pulse.port_group) -> !pulse.port


module {
func @x0(%arg0: !pulse.port_group) {
%0 = "pulse.select_port"(%arg0) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%cst = arith.constant 0.17980737117787168 : f64
%cst_0 = arith.constant 0.000000e+00 : f64
%1 = complex.create %cst, %cst_0 : complex<f64>
%c40_i32 = arith.constant 40 : i32
%cst_1 = arith.constant 0.6999359173856885 : f64
%c160_i32 = arith.constant 160 : i32
%2 = pulse.drag(%c160_i32, %1, %c40_i32, %cst_1) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%0, %2) : (!pulse.port, !pulse.waveform)
return
}
func @x1(%arg0: !pulse.port_group) {
%0 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst = arith.constant 0.16946846376475716 : f64
%cst_0 = arith.constant 0.000000e+00 : f64
%1 = complex.create %cst, %cst_0 : complex<f64>
%c40_i32 = arith.constant 40 : i32
%cst_1 = arith.constant -0.68671951029591838 : f64
%c160_i32 = arith.constant 160 : i32
%2 = pulse.drag(%c160_i32, %1, %c40_i32, %cst_1) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%0, %2) : (!pulse.port, !pulse.waveform)
return
}
func @x2(%arg0: !pulse.port_group) {
%0 = "pulse.select_port"(%arg0) {id = "d2"} : (!pulse.port_group) -> !pulse.port
%cst = arith.constant 0.18917049559967117 : f64
%cst_0 = arith.constant 0.000000e+00 : f64
%1 = complex.create %cst, %cst_0 : complex<f64>
%c40_i32 = arith.constant 40 : i32
%cst_1 = arith.constant 0.75794626434693213 : f64
%c160_i32 = arith.constant 160 : i32
%2 = pulse.drag(%c160_i32, %1, %c40_i32, %cst_1) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%0, %2) : (!pulse.port, !pulse.waveform)
return
}
func @x3(%arg0: !pulse.port_group) {
%0 = "pulse.select_port"(%arg0) {id = "d3"} : (!pulse.port_group) -> !pulse.port
%cst = arith.constant 0.19038039184667085 : f64
%cst_0 = arith.constant 0.000000e+00 : f64
%1 = complex.create %cst, %cst_0 : complex<f64>
%c40_i32 = arith.constant 40 : i32
%cst_1 = arith.constant 0.74061073242105191 : f64
%c160_i32 = arith.constant 160 : i32
%2 = pulse.drag(%c160_i32, %1, %c40_i32, %cst_1) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%0, %2) : (!pulse.port, !pulse.waveform)
return
}
func @cx01(%arg0: !pulse.port_group, %arg1: !pulse.port_group) {
%0 = "pulse.select_port"(%arg0) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%cst = arith.constant 1.5707963267948966 : f64
pulse.shift_phase(%0, %cst) : (!pulse.port, f64)
%1 = "pulse.select_port"(%arg0) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%cst_0 = arith.constant -3.3030078236412091E-17 : f64
%cst_1 = arith.constant -0.17980737117787168 : f64
%2 = complex.create %cst_0, %cst_1 : complex<f64>
%c40_i32 = arith.constant 40 : i32
%cst_2 = arith.constant 0.6999359173856885 : f64
%c160_i32 = arith.constant 160 : i32
%3 = pulse.drag(%c160_i32, %2, %c40_i32, %cst_2) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%1, %3) : (!pulse.port, !pulse.waveform)
%4 = "pulse.select_port"(%arg0) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%c672_i32 = arith.constant 672 : i32
pulse.delay(%c672_i32, %4) : (i32, !pulse.port)
%5 = "pulse.select_port"(%arg0) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%cst_3 = arith.constant 0.17980737117787168 : f64
%cst_4 = arith.constant 0.000000e+00 : f64
%6 = complex.create %cst_3, %cst_4 : complex<f64>
%c40_i32_5 = arith.constant 40 : i32
%cst_6 = arith.constant 0.6999359173856885 : f64
%c160_i32_7 = arith.constant 160 : i32
%7 = pulse.drag(%c160_i32_7, %6, %c40_i32_5, %cst_6) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%5, %7) : (!pulse.port, !pulse.waveform)
%8 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_8 = arith.constant 0.084483000576426853 : f64
%cst_9 = arith.constant 5.1781053983668951E-4 : f64
%9 = complex.create %cst_8, %cst_9 : complex<f64>
%c40_i32_10 = arith.constant 40 : i32
%cst_11 = arith.constant -0.63693584079030685 : f64
%c160_i32_12 = arith.constant 160 : i32
%10 = pulse.drag(%c160_i32_12, %9, %c40_i32_10, %cst_11) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%8, %10) : (!pulse.port, !pulse.waveform)
%11 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%c160_i32_13 = arith.constant 160 : i32
pulse.delay(%c160_i32_13, %11) : (i32, !pulse.port)
%12 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_14 = arith.constant 0.061110783182318495 : f64
%cst_15 = arith.constant -1.5531471574357976E-4 : f64
%13 = complex.create %cst_14, %cst_15 : complex<f64>
%c64_i32 = arith.constant 64 : i32
%c256_i32 = arith.constant 256 : i32
%c512_i32 = arith.constant 512 : i32
%14 = pulse.gaussian_square(%c512_i32, %13, %c64_i32, %c256_i32) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%12, %14) : (!pulse.port, !pulse.waveform)
%15 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%c832_i32 = arith.constant 832 : i32
pulse.delay(%c832_i32, %15) : (i32, !pulse.port)
%16 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_16 = arith.constant -0.061110783182318495 : f64
%cst_17 = arith.constant 1.5531471574358724E-4 : f64
%17 = complex.create %cst_16, %cst_17 : complex<f64>
%c64_i32_18 = arith.constant 64 : i32
%c256_i32_19 = arith.constant 256 : i32
%c512_i32_20 = arith.constant 512 : i32
%18 = pulse.gaussian_square(%c512_i32_20, %17, %c64_i32_18, %c256_i32_19) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%16, %18) : (!pulse.port, !pulse.waveform)
%19 = "pulse.select_port"(%arg0) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%c160_i32_21 = arith.constant 160 : i32
pulse.delay(%c160_i32_21, %19) : (i32, !pulse.port)
%20 = "pulse.select_port"(%arg0) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%cst_22 = arith.constant -0.22819308603366056 : f64
%cst_23 = arith.constant -0.32518251289747546 : f64
%21 = complex.create %cst_22, %cst_23 : complex<f64>
%c64_i32_24 = arith.constant 64 : i32
%c256_i32_25 = arith.constant 256 : i32
%c512_i32_26 = arith.constant 512 : i32
%22 = pulse.gaussian_square(%c512_i32_26, %21, %c64_i32_24, %c256_i32_25) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%20, %22) : (!pulse.port, !pulse.waveform)
%23 = "pulse.select_port"(%arg0) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%c832_i32_27 = arith.constant 832 : i32
pulse.delay(%c832_i32_27, %23) : (i32, !pulse.port)
%24 = "pulse.select_port"(%arg0) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%cst_28 = arith.constant 0.22819308603366059 : f64
%cst_29 = arith.constant 0.32518251289747541 : f64
%25 = complex.create %cst_28, %cst_29 : complex<f64>
%c64_i32_30 = arith.constant 64 : i32
%c256_i32_31 = arith.constant 256 : i32
%c512_i32_32 = arith.constant 512 : i32
%26 = pulse.gaussian_square(%c512_i32_32, %25, %c64_i32_30, %c256_i32_31) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%24, %26) : (!pulse.port, !pulse.waveform)
%27 = "pulse.select_port"(%arg0) {id = "u1"} : (!pulse.port_group) -> !pulse.port
%cst_33 = arith.constant 1.5707963267948966 : f64
pulse.shift_phase(%27, %cst_33) : (!pulse.port, f64)
return
}
func @cx10(%arg0: !pulse.port_group, %arg1: !pulse.port_group) {
%0 = "pulse.select_port"(%arg0) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%cst = arith.constant -1.5707963267948966 : f64
pulse.shift_phase(%0, %cst) : (!pulse.port, f64)
%1 = "pulse.select_port"(%arg0) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%cst_0 = arith.constant 0.089776566073944741 : f64
%cst_1 = arith.constant -3.3436108695185151E-4 : f64
%2 = complex.create %cst_0, %cst_1 : complex<f64>
%c40_i32 = arith.constant 40 : i32
%cst_2 = arith.constant 0.74011253340348393 : f64
%c160_i32 = arith.constant 160 : i32
%3 = pulse.drag(%c160_i32, %2, %c40_i32, %cst_2) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%1, %3) : (!pulse.port, !pulse.waveform)
%4 = "pulse.select_port"(%arg0) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%c672_i32 = arith.constant 672 : i32
pulse.delay(%c672_i32, %4) : (i32, !pulse.port)
%5 = "pulse.select_port"(%arg0) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%cst_3 = arith.constant 0.17980737117787168 : f64
%cst_4 = arith.constant 0.000000e+00 : f64
%6 = complex.create %cst_3, %cst_4 : complex<f64>
%c40_i32_5 = arith.constant 40 : i32
%cst_6 = arith.constant 0.6999359173856885 : f64
%c160_i32_7 = arith.constant 160 : i32
%7 = pulse.drag(%c160_i32_7, %6, %c40_i32_5, %cst_6) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%5, %7) : (!pulse.port, !pulse.waveform)
%8 = "pulse.select_port"(%arg0) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%c1344_i32 = arith.constant 1344 : i32
pulse.delay(%c1344_i32, %8) : (i32, !pulse.port)
%9 = "pulse.select_port"(%arg0) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%cst_8 = arith.constant -3.3436108695188198E-4 : f64
%cst_9 = arith.constant -0.089776566073944741 : f64
%10 = complex.create %cst_8, %cst_9 : complex<f64>
%c40_i32_10 = arith.constant 40 : i32
%cst_11 = arith.constant 0.74011253340348393 : f64
%c160_i32_12 = arith.constant 160 : i32
%11 = pulse.drag(%c160_i32_12, %10, %c40_i32_10, %cst_11) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%9, %11) : (!pulse.port, !pulse.waveform)
%12 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_13 = arith.constant -3.1415926535897931 : f64
pulse.shift_phase(%12, %cst_13) : (!pulse.port, f64)
%13 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_14 = arith.constant -5.1781053983669201E-4 : f64
%cst_15 = arith.constant 0.084483000576426853 : f64
%14 = complex.create %cst_14, %cst_15 : complex<f64>
%c40_i32_16 = arith.constant 40 : i32
%cst_17 = arith.constant -0.63693584079030685 : f64
%c160_i32_18 = arith.constant 160 : i32
%15 = pulse.drag(%c160_i32_18, %14, %c40_i32_16, %cst_17) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%13, %15) : (!pulse.port, !pulse.waveform)
%16 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%c160_i32_19 = arith.constant 160 : i32
pulse.delay(%c160_i32_19, %16) : (i32, !pulse.port)
%17 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_20 = arith.constant 0.061110783182318495 : f64
%cst_21 = arith.constant -1.5531471574357976E-4 : f64
%18 = complex.create %cst_20, %cst_21 : complex<f64>
%c64_i32 = arith.constant 64 : i32
%c256_i32 = arith.constant 256 : i32
%c512_i32 = arith.constant 512 : i32
%19 = pulse.gaussian_square(%c512_i32, %18, %c64_i32, %c256_i32) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%17, %19) : (!pulse.port, !pulse.waveform)
%20 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%c832_i32 = arith.constant 832 : i32
pulse.delay(%c832_i32, %20) : (i32, !pulse.port)
%21 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_22 = arith.constant -0.061110783182318495 : f64
%cst_23 = arith.constant 1.5531471574358724E-4 : f64
%22 = complex.create %cst_22, %cst_23 : complex<f64>
%c64_i32_24 = arith.constant 64 : i32
%c256_i32_25 = arith.constant 256 : i32
%c512_i32_26 = arith.constant 512 : i32
%23 = pulse.gaussian_square(%c512_i32_26, %22, %c64_i32_24, %c256_i32_25) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%21, %23) : (!pulse.port, !pulse.waveform)
%24 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%c1344_i32_27 = arith.constant 1344 : i32
pulse.delay(%c1344_i32_27, %24) : (i32, !pulse.port)
%25 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_28 = arith.constant -1.5707963267948966 : f64
pulse.shift_phase(%25, %cst_28) : (!pulse.port, f64)
%26 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%c1344_i32_29 = arith.constant 1344 : i32
pulse.delay(%c1344_i32_29, %26) : (i32, !pulse.port)
%27 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_30 = arith.constant 0.084483000576426853 : f64
%cst_31 = arith.constant 5.1781053983668951E-4 : f64
%28 = complex.create %cst_30, %cst_31 : complex<f64>
%c40_i32_32 = arith.constant 40 : i32
%cst_33 = arith.constant -0.63693584079030685 : f64
%c160_i32_34 = arith.constant 160 : i32
%29 = pulse.drag(%c160_i32_34, %28, %c40_i32_32, %cst_33) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%27, %29) : (!pulse.port, !pulse.waveform)
%30 = "pulse.select_port"(%arg0) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%cst_35 = arith.constant -3.1415926535897931 : f64
pulse.shift_phase(%30, %cst_35) : (!pulse.port, f64)
%31 = "pulse.select_port"(%arg0) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%c160_i32_36 = arith.constant 160 : i32
pulse.delay(%c160_i32_36, %31) : (i32, !pulse.port)
%32 = "pulse.select_port"(%arg0) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%cst_37 = arith.constant -0.22819308603366056 : f64
%cst_38 = arith.constant -0.32518251289747546 : f64
%33 = complex.create %cst_37, %cst_38 : complex<f64>
%c64_i32_39 = arith.constant 64 : i32
%c256_i32_40 = arith.constant 256 : i32
%c512_i32_41 = arith.constant 512 : i32
%34 = pulse.gaussian_square(%c512_i32_41, %33, %c64_i32_39, %c256_i32_40) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%32, %34) : (!pulse.port, !pulse.waveform)
%35 = "pulse.select_port"(%arg0) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%c832_i32_42 = arith.constant 832 : i32
pulse.delay(%c832_i32_42, %35) : (i32, !pulse.port)
%36 = "pulse.select_port"(%arg0) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%cst_43 = arith.constant 0.22819308603366059 : f64
%cst_44 = arith.constant 0.32518251289747541 : f64
%37 = complex.create %cst_43, %cst_44 : complex<f64>
%c64_i32_45 = arith.constant 64 : i32
%c256_i32_46 = arith.constant 256 : i32
%c512_i32_47 = arith.constant 512 : i32
%38 = pulse.gaussian_square(%c512_i32_47, %37, %c64_i32_45, %c256_i32_46) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%36, %38) : (!pulse.port, !pulse.waveform)
%39 = "pulse.select_port"(%arg0) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%c1344_i32_48 = arith.constant 1344 : i32
pulse.delay(%c1344_i32_48, %39) : (i32, !pulse.port)
%40 = "pulse.select_port"(%arg0) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%cst_49 = arith.constant -1.5707963267948966 : f64
pulse.shift_phase(%40, %cst_49) : (!pulse.port, f64)
%41 = "pulse.select_port"(%arg0) {id = "u1"} : (!pulse.port_group) -> !pulse.port
%cst_50 = arith.constant -1.5707963267948966 : f64
pulse.shift_phase(%41, %cst_50) : (!pulse.port, f64)
%42 = "pulse.select_port"(%arg0) {id = "u4"} : (!pulse.port_group) -> !pulse.port
%cst_51 = arith.constant -3.1415926535897931 : f64
pulse.shift_phase(%42, %cst_51) : (!pulse.port, f64)
%43 = "pulse.select_port"(%arg0) {id = "u4"} : (!pulse.port_group) -> !pulse.port
%c1344_i32_52 = arith.constant 1344 : i32
pulse.delay(%c1344_i32_52, %43) : (i32, !pulse.port)
%44 = "pulse.select_port"(%arg0) {id = "u4"} : (!pulse.port_group) -> !pulse.port
%cst_53 = arith.constant -1.5707963267948966 : f64
pulse.shift_phase(%44, %cst_53) : (!pulse.port, f64)
%45 = "pulse.select_port"(%arg0) {id = "u8"} : (!pulse.port_group) -> !pulse.port
%cst_54 = arith.constant -3.1415926535897931 : f64
pulse.shift_phase(%45, %cst_54) : (!pulse.port, f64)
%46 = "pulse.select_port"(%arg0) {id = "u8"} : (!pulse.port_group) -> !pulse.port
%c1344_i32_55 = arith.constant 1344 : i32
pulse.delay(%c1344_i32_55, %46) : (i32, !pulse.port)
%47 = "pulse.select_port"(%arg0) {id = "u8"} : (!pulse.port_group) -> !pulse.port
%cst_56 = arith.constant -1.5707963267948966 : f64
pulse.shift_phase(%47, %cst_56) : (!pulse.port, f64)
return
}
func @cx12(%arg0: !pulse.port_group, %arg1: !pulse.port_group) {
%0 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst = arith.constant 1.5707963267948966 : f64
pulse.shift_phase(%0, %cst) : (!pulse.port, f64)
%1 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_0 = arith.constant -3.1130851755889356E-17 : f64
%cst_1 = arith.constant -0.16946846376475716 : f64
%2 = complex.create %cst_0, %cst_1 : complex<f64>
%c40_i32 = arith.constant 40 : i32
%cst_2 = arith.constant -0.68671951029591838 : f64
%c160_i32 = arith.constant 160 : i32
%3 = pulse.drag(%c160_i32, %2, %c40_i32, %cst_2) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%1, %3) : (!pulse.port, !pulse.waveform)
%4 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%c912_i32 = arith.constant 912 : i32
pulse.delay(%c912_i32, %4) : (i32, !pulse.port)
%5 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_3 = arith.constant 0.16946846376475716 : f64
%cst_4 = arith.constant 0.000000e+00 : f64
%6 = complex.create %cst_3, %cst_4 : complex<f64>
%c40_i32_5 = arith.constant 40 : i32
%cst_6 = arith.constant -0.68671951029591838 : f64
%c160_i32_7 = arith.constant 160 : i32
%7 = pulse.drag(%c160_i32_7, %6, %c40_i32_5, %cst_6) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%5, %7) : (!pulse.port, !pulse.waveform)
%8 = "pulse.select_port"(%arg0) {id = "d2"} : (!pulse.port_group) -> !pulse.port
%cst_8 = arith.constant 0.094540487800318596 : f64
%cst_9 = arith.constant -3.9710888891587696E-4 : f64
%9 = complex.create %cst_8, %cst_9 : complex<f64>
%c40_i32_10 = arith.constant 40 : i32
%cst_11 = arith.constant 0.85354566868785298 : f64
%c160_i32_12 = arith.constant 160 : i32
%10 = pulse.drag(%c160_i32_12, %9, %c40_i32_10, %cst_11) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%8, %10) : (!pulse.port, !pulse.waveform)
%11 = "pulse.select_port"(%arg0) {id = "d2"} : (!pulse.port_group) -> !pulse.port
%c160_i32_13 = arith.constant 160 : i32
pulse.delay(%c160_i32_13, %11) : (i32, !pulse.port)
%12 = "pulse.select_port"(%arg0) {id = "d2"} : (!pulse.port_group) -> !pulse.port
%cst_14 = arith.constant 0.039131757733167315 : f64
%cst_15 = arith.constant -6.663466510650446E-4 : f64
%13 = complex.create %cst_14, %cst_15 : complex<f64>
%c64_i32 = arith.constant 64 : i32
%c496_i32 = arith.constant 496 : i32
%c752_i32 = arith.constant 752 : i32
%14 = pulse.gaussian_square(%c752_i32, %13, %c64_i32, %c496_i32) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%12, %14) : (!pulse.port, !pulse.waveform)
%15 = "pulse.select_port"(%arg0) {id = "d2"} : (!pulse.port_group) -> !pulse.port
%c1072_i32 = arith.constant 1072 : i32
pulse.delay(%c1072_i32, %15) : (i32, !pulse.port)
%16 = "pulse.select_port"(%arg0) {id = "d2"} : (!pulse.port_group) -> !pulse.port
%cst_16 = arith.constant -0.039131757733167315 : f64
%cst_17 = arith.constant 6.6634665106504937E-4 : f64
%17 = complex.create %cst_16, %cst_17 : complex<f64>
%c64_i32_18 = arith.constant 64 : i32
%c496_i32_19 = arith.constant 496 : i32
%c752_i32_20 = arith.constant 752 : i32
%18 = pulse.gaussian_square(%c752_i32_20, %17, %c64_i32_18, %c496_i32_19) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%16, %18) : (!pulse.port, !pulse.waveform)
%19 = "pulse.select_port"(%arg0) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%cst_21 = arith.constant 1.5707963267948966 : f64
pulse.shift_phase(%19, %cst_21) : (!pulse.port, f64)
%20 = "pulse.select_port"(%arg0) {id = "u2"} : (!pulse.port_group) -> !pulse.port
%c160_i32_22 = arith.constant 160 : i32
pulse.delay(%c160_i32_22, %20) : (i32, !pulse.port)
%21 = "pulse.select_port"(%arg0) {id = "u2"} : (!pulse.port_group) -> !pulse.port
%cst_23 = arith.constant -0.4672295345405707 : f64
%cst_24 = arith.constant -0.010192806263952648 : f64
%22 = complex.create %cst_23, %cst_24 : complex<f64>
%c64_i32_25 = arith.constant 64 : i32
%c496_i32_26 = arith.constant 496 : i32
%c752_i32_27 = arith.constant 752 : i32
%23 = pulse.gaussian_square(%c752_i32_27, %22, %c64_i32_25, %c496_i32_26) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%21, %23) : (!pulse.port, !pulse.waveform)
%24 = "pulse.select_port"(%arg0) {id = "u2"} : (!pulse.port_group) -> !pulse.port
%c1072_i32_28 = arith.constant 1072 : i32
pulse.delay(%c1072_i32_28, %24) : (i32, !pulse.port)
%25 = "pulse.select_port"(%arg0) {id = "u2"} : (!pulse.port_group) -> !pulse.port
%cst_29 = arith.constant 0.4672295345405707 : f64
%cst_30 = arith.constant 0.010192806263952591 : f64
%26 = complex.create %cst_29, %cst_30 : complex<f64>
%c64_i32_31 = arith.constant 64 : i32
%c496_i32_32 = arith.constant 496 : i32
%c752_i32_33 = arith.constant 752 : i32
%27 = pulse.gaussian_square(%c752_i32_33, %26, %c64_i32_31, %c496_i32_32) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%25, %27) : (!pulse.port, !pulse.waveform)
%28 = "pulse.select_port"(%arg0) {id = "u4"} : (!pulse.port_group) -> !pulse.port
%cst_34 = arith.constant 1.5707963267948966 : f64
pulse.shift_phase(%28, %cst_34) : (!pulse.port, f64)
%29 = "pulse.select_port"(%arg0) {id = "u8"} : (!pulse.port_group) -> !pulse.port
%cst_35 = arith.constant 1.5707963267948966 : f64
pulse.shift_phase(%29, %cst_35) : (!pulse.port, f64)
return
}
func @rz0(%arg0: !pulse.port_group, %arg1: !quir.angle<10>) {
%0 = "pulse.select_port"(%arg0) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%cst = arith.constant 0.000000e+00 : f64
pulse.shift_phase(%0, %cst) : (!pulse.port, f64)
%1 = "pulse.select_port"(%arg0) {id = "u1"} : (!pulse.port_group) -> !pulse.port
%cst_0 = arith.constant 0.000000e+00 : f64
pulse.shift_phase(%1, %cst_0) : (!pulse.port, f64)
return
}
func @rz1(%arg0: !pulse.port_group, %arg1: !quir.angle<10>) {
%0 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst = arith.constant 0.000000e+00 : f64
pulse.shift_phase(%0, %cst) : (!pulse.port, f64)
%1 = "pulse.select_port"(%arg0) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%cst_0 = arith.constant 0.000000e+00 : f64
pulse.shift_phase(%1, %cst_0) : (!pulse.port, f64)
%2 = "pulse.select_port"(%arg0) {id = "u4"} : (!pulse.port_group) -> !pulse.port
%cst_1 = arith.constant 0.000000e+00 : f64
pulse.shift_phase(%2, %cst_1) : (!pulse.port, f64)
%3 = "pulse.select_port"(%arg0) {id = "u8"} : (!pulse.port_group) -> !pulse.port
%cst_2 = arith.constant 0.000000e+00 : f64
pulse.shift_phase(%3, %cst_2) : (!pulse.port, f64)
return
}
func @sx0(%arg0: !pulse.port_group) {
%0 = "pulse.select_port"(%arg0) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%cst = arith.constant 0.089776566073944741 : f64
%cst_0 = arith.constant -3.3436108695185151E-4 : f64
%1 = complex.create %cst, %cst_0 : complex<f64>
%c40_i32 = arith.constant 40 : i32
%cst_1 = arith.constant 0.74011253340348393 : f64
%c160_i32 = arith.constant 160 : i32
%2 = pulse.drag(%c160_i32, %1, %c40_i32, %cst_1) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%0, %2) : (!pulse.port, !pulse.waveform)
return
}
func @sx1(%arg0: !pulse.port_group) {
%0 = "pulse.select_port"(%arg0) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst = arith.constant 0.084483000576426853 : f64
%cst_0 = arith.constant 5.1781053983668951E-4 : f64
%1 = complex.create %cst, %cst_0 : complex<f64>
%c40_i32 = arith.constant 40 : i32
%cst_1 = arith.constant -0.63693584079030685 : f64
%c160_i32 = arith.constant 160 : i32
%2 = pulse.drag(%c160_i32, %1, %c40_i32, %cst_1) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%0, %2) : (!pulse.port, !pulse.waveform)
return
}
func @sx2(%arg0: !pulse.port_group) {
%0 = "pulse.select_port"(%arg0) {id = "d2"} : (!pulse.port_group) -> !pulse.port
%cst = arith.constant 0.094540487800318596 : f64
%cst_0 = arith.constant -3.9710888891587696E-4 : f64
%1 = complex.create %cst, %cst_0 : complex<f64>
%c40_i32 = arith.constant 40 : i32
%cst_1 = arith.constant 0.85354566868785298 : f64
%c160_i32 = arith.constant 160 : i32
%2 = pulse.drag(%c160_i32, %1, %c40_i32, %cst_1) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%0, %2) : (!pulse.port, !pulse.waveform)
return
}
func @sx3(%arg0: !pulse.port_group) {
%0 = "pulse.select_port"(%arg0) {id = "d3"} : (!pulse.port_group) -> !pulse.port
%cst = arith.constant 0.095293610036729287 : f64
%cst_0 = arith.constant 0.0010061143295647052 : f64
%1 = complex.create %cst, %cst_0 : complex<f64>
%c40_i32 = arith.constant 40 : i32
%cst_1 = arith.constant 0.74995439781122553 : f64
%c160_i32 = arith.constant 160 : i32
%2 = pulse.drag(%c160_i32, %1, %c40_i32, %cst_1) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%0, %2) : (!pulse.port, !pulse.waveform)
return
}
func @main() -> i32 {
%0 = "pulse.create_port"() {uid = "d0"} : () -> !pulse.port
%1 = "pulse.create_port"() {uid = "d1"} : () -> !pulse.port
%2 = "pulse.create_port"() {uid = "u0"} : () -> !pulse.port
%3 = "pulse.create_port"() {uid = "u1"} : () -> !pulse.port
%4 = "pulse.create_port"() {uid = "u4"} : () -> !pulse.port
%5 = "pulse.create_port"() {uid = "u8"} : () -> !pulse.port
%6 = "pulse.create_port"() {uid = "m0"} : () -> !pulse.port
%7 = "pulse.create_port"() {uid = "acquire"} : () -> !pulse.port
%8 = "pulse.create_port"() {uid = "m1"} : () -> !pulse.port
%9 = "pulse.create_port"() {uid = "m10"} : () -> !pulse.port
%10 = "pulse.create_port"() {uid = "m11"} : () -> !pulse.port
%11 = "pulse.create_port"() {uid = "m12"} : () -> !pulse.port
%12 = "pulse.create_port"() {uid = "m13"} : () -> !pulse.port
%13 = "pulse.create_port"() {uid = "m14"} : () -> !pulse.port
%14 = "pulse.create_port"() {uid = "m15"} : () -> !pulse.port
%15 = "pulse.create_port"() {uid = "m16"} : () -> !pulse.port
%16 = "pulse.create_port"() {uid = "m17"} : () -> !pulse.port
%17 = "pulse.create_port"() {uid = "m18"} : () -> !pulse.port
%18 = "pulse.create_port"() {uid = "m19"} : () -> !pulse.port
%19 = "pulse.create_port"() {uid = "m2"} : () -> !pulse.port
%20 = "pulse.create_port"() {uid = "m20"} : () -> !pulse.port
%21 = "pulse.create_port"() {uid = "m21"} : () -> !pulse.port
%22 = "pulse.create_port"() {uid = "m22"} : () -> !pulse.port
%23 = "pulse.create_port"() {uid = "m23"} : () -> !pulse.port
%24 = "pulse.create_port"() {uid = "m24"} : () -> !pulse.port
%25 = "pulse.create_port"() {uid = "m25"} : () -> !pulse.port
%26 = "pulse.create_port"() {uid = "m26"} : () -> !pulse.port
%27 = "pulse.create_port"() {uid = "m3"} : () -> !pulse.port
%28 = "pulse.create_port"() {uid = "m4"} : () -> !pulse.port
%29 = "pulse.create_port"() {uid = "m5"} : () -> !pulse.port
%30 = "pulse.create_port"() {uid = "m6"} : () -> !pulse.port
%31 = "pulse.create_port"() {uid = "m7"} : () -> !pulse.port
%32 = "pulse.create_port"() {uid = "m8"} : () -> !pulse.port
%33 = "pulse.create_port"() {uid = "m9"} : () -> !pulse.port
%34 = "pulse.create_port_group"(%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %7) : (!pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port) -> !pulse.port_group
%35 = "pulse.create_port"() {uid = "d2"} : () -> !pulse.port
%36 = "pulse.create_port"() {uid = "u2"} : () -> !pulse.port
%37 = "pulse.create_port"() {uid = "d4"} : () -> !pulse.port
%38 = "pulse.create_port"() {uid = "u13"} : () -> !pulse.port
%39 = "pulse.create_port"() {uid = "u3"} : () -> !pulse.port
%40 = "pulse.create_port"() {uid = "u6"} : () -> !pulse.port
%41 = "pulse.create_port_group"(%0, %1, %2, %3, %4, %5, %35, %36, %37, %38, %39, %40, %6, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %7, %7) : (!pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port) -> !pulse.port_group
%42 = "pulse.create_port"() {uid = "d3"} : () -> !pulse.port
%43 = "pulse.create_port"() {uid = "u5"} : () -> !pulse.port
%44 = "pulse.create_port"() {uid = "u10"} : () -> !pulse.port
%45 = "pulse.create_port_group"(%1, %35, %2, %36, %4, %5, %40, %42, %43, %44, %6, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %7, %7) : (!pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port) -> !pulse.port_group
%46 = "pulse.create_port"() {uid = "d5"} : () -> !pulse.port
%47 = "pulse.create_port"() {uid = "u16"} : () -> !pulse.port
%48 = "pulse.create_port"() {uid = "u7"} : () -> !pulse.port
%49 = "pulse.create_port_group"(%35, %42, %36, %43, %40, %44, %46, %47, %48, %6, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %7, %7) : (!pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port, !pulse.port) -> !pulse.port_group
call @x0(%34) : (!pulse.port_group) -> ()
call @x1(%41) : (!pulse.port_group) -> ()
call @x2(%45) : (!pulse.port_group) -> ()
call @x3(%49) : (!pulse.port_group) -> ()
call @cx01(%34, %41) : (!pulse.port_group, !pulse.port_group) -> ()
call @cx10(%41, %34) : (!pulse.port_group, !pulse.port_group) -> ()
call @cx12(%41, %45) : (!pulse.port_group, !pulse.port_group) -> ()
%50 = quir.constant #quir.angle<3.140000e+00 : !quir.angle<10>>
%51 = quir.constant #quir.angle<1.070000e+00 : !quir.angle<10>>
%52 = quir.constant #quir.angle<5.350000e-01 : !quir.angle<10>>
call @rz0(%34, %50) : (!pulse.port_group, !quir.angle<10>) -> ()
call @rz1(%41, %51) : (!pulse.port_group, !quir.angle<10>) -> ()
call @sx0(%34) : (!pulse.port_group) -> ()
call @sx1(%41) : (!pulse.port_group) -> ()
call @sx2(%45) : (!pulse.port_group) -> ()
call @sx3(%49) : (!pulse.port_group) -> ()
%c0_i32 = arith.constant 0 : i32
return %c0_i32 : i32
}
}
