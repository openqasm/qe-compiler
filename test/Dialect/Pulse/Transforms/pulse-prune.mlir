// RUN: qss-compiler -X=mlir --pulse-prune %s | FileCheck %s
// XFAIL: *

// CHECK-NOT:  "pulse.create_port_group"
// CHECK-NOT:  "pulse.select_port"

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
%50 = "pulse.select_port"(%34) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%cst = arith.constant 0.17980737117787168 : f64
%cst_0 = arith.constant 0.000000e+00 : f64
%51 = complex.create %cst, %cst_0 : complex<f64>
%c40_i32 = arith.constant 40 : i32
%cst_1 = arith.constant 0.6999359173856885 : f64
%c160_i32 = arith.constant 160 : i32
%52 = pulse.drag(%c160_i32, %51, %c40_i32, %cst_1) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%50, %52) : (!pulse.port, !pulse.waveform)
%53 = "pulse.select_port"(%41) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_2 = arith.constant 0.16946846376475716 : f64
%cst_3 = arith.constant 0.000000e+00 : f64
%54 = complex.create %cst_2, %cst_3 : complex<f64>
%c40_i32_4 = arith.constant 40 : i32
%cst_5 = arith.constant -0.68671951029591838 : f64
%c160_i32_6 = arith.constant 160 : i32
%55 = pulse.drag(%c160_i32_6, %54, %c40_i32_4, %cst_5) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%53, %55) : (!pulse.port, !pulse.waveform)
%56 = "pulse.select_port"(%45) {id = "d2"} : (!pulse.port_group) -> !pulse.port
%cst_7 = arith.constant 0.18917049559967117 : f64
%cst_8 = arith.constant 0.000000e+00 : f64
%57 = complex.create %cst_7, %cst_8 : complex<f64>
%c40_i32_9 = arith.constant 40 : i32
%cst_10 = arith.constant 0.75794626434693213 : f64
%c160_i32_11 = arith.constant 160 : i32
%58 = pulse.drag(%c160_i32_11, %57, %c40_i32_9, %cst_10) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%56, %58) : (!pulse.port, !pulse.waveform)
%59 = "pulse.select_port"(%49) {id = "d3"} : (!pulse.port_group) -> !pulse.port
%cst_12 = arith.constant 0.19038039184667085 : f64
%cst_13 = arith.constant 0.000000e+00 : f64
%60 = complex.create %cst_12, %cst_13 : complex<f64>
%c40_i32_14 = arith.constant 40 : i32
%cst_15 = arith.constant 0.74061073242105191 : f64
%c160_i32_16 = arith.constant 160 : i32
%61 = pulse.drag(%c160_i32_16, %60, %c40_i32_14, %cst_15) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%59, %61) : (!pulse.port, !pulse.waveform)
%62 = "pulse.select_port"(%34) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%cst_17 = arith.constant 1.5707963267948966 : f64
pulse.shift_phase(%62, %cst_17) : (!pulse.port, f64)
%63 = "pulse.select_port"(%34) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%cst_18 = arith.constant -3.3030078236412091E-17 : f64
%cst_19 = arith.constant -0.17980737117787168 : f64
%64 = complex.create %cst_18, %cst_19 : complex<f64>
%c40_i32_20 = arith.constant 40 : i32
%cst_21 = arith.constant 0.6999359173856885 : f64
%c160_i32_22 = arith.constant 160 : i32
%65 = pulse.drag(%c160_i32_22, %64, %c40_i32_20, %cst_21) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%63, %65) : (!pulse.port, !pulse.waveform)
%66 = "pulse.select_port"(%34) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%c672_i32 = arith.constant 672 : i32
pulse.delay(%c672_i32, %66) : (i32, !pulse.port)
%67 = "pulse.select_port"(%34) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%cst_23 = arith.constant 0.17980737117787168 : f64
%cst_24 = arith.constant 0.000000e+00 : f64
%68 = complex.create %cst_23, %cst_24 : complex<f64>
%c40_i32_25 = arith.constant 40 : i32
%cst_26 = arith.constant 0.6999359173856885 : f64
%c160_i32_27 = arith.constant 160 : i32
%69 = pulse.drag(%c160_i32_27, %68, %c40_i32_25, %cst_26) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%67, %69) : (!pulse.port, !pulse.waveform)
%70 = "pulse.select_port"(%34) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_28 = arith.constant 0.084483000576426853 : f64
%cst_29 = arith.constant 5.1781053983668951E-4 : f64
%71 = complex.create %cst_28, %cst_29 : complex<f64>
%c40_i32_30 = arith.constant 40 : i32
%cst_31 = arith.constant -0.63693584079030685 : f64
%c160_i32_32 = arith.constant 160 : i32
%72 = pulse.drag(%c160_i32_32, %71, %c40_i32_30, %cst_31) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%70, %72) : (!pulse.port, !pulse.waveform)
%73 = "pulse.select_port"(%34) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%c160_i32_33 = arith.constant 160 : i32
pulse.delay(%c160_i32_33, %73) : (i32, !pulse.port)
%74 = "pulse.select_port"(%34) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_34 = arith.constant 0.061110783182318495 : f64
%cst_35 = arith.constant -1.5531471574357976E-4 : f64
%75 = complex.create %cst_34, %cst_35 : complex<f64>
%c64_i32 = arith.constant 64 : i32
%c256_i32 = arith.constant 256 : i32
%c512_i32 = arith.constant 512 : i32
%76 = pulse.gaussian_square(%c512_i32, %75, %c64_i32, %c256_i32) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%74, %76) : (!pulse.port, !pulse.waveform)
%77 = "pulse.select_port"(%34) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%c832_i32 = arith.constant 832 : i32
pulse.delay(%c832_i32, %77) : (i32, !pulse.port)
%78 = "pulse.select_port"(%34) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_36 = arith.constant -0.061110783182318495 : f64
%cst_37 = arith.constant 1.5531471574358724E-4 : f64
%79 = complex.create %cst_36, %cst_37 : complex<f64>
%c64_i32_38 = arith.constant 64 : i32
%c256_i32_39 = arith.constant 256 : i32
%c512_i32_40 = arith.constant 512 : i32
%80 = pulse.gaussian_square(%c512_i32_40, %79, %c64_i32_38, %c256_i32_39) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%78, %80) : (!pulse.port, !pulse.waveform)
%81 = "pulse.select_port"(%34) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%c160_i32_41 = arith.constant 160 : i32
pulse.delay(%c160_i32_41, %81) : (i32, !pulse.port)
%82 = "pulse.select_port"(%34) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%cst_42 = arith.constant -0.22819308603366056 : f64
%cst_43 = arith.constant -0.32518251289747546 : f64
%83 = complex.create %cst_42, %cst_43 : complex<f64>
%c64_i32_44 = arith.constant 64 : i32
%c256_i32_45 = arith.constant 256 : i32
%c512_i32_46 = arith.constant 512 : i32
%84 = pulse.gaussian_square(%c512_i32_46, %83, %c64_i32_44, %c256_i32_45) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%82, %84) : (!pulse.port, !pulse.waveform)
%85 = "pulse.select_port"(%34) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%c832_i32_47 = arith.constant 832 : i32
pulse.delay(%c832_i32_47, %85) : (i32, !pulse.port)
%86 = "pulse.select_port"(%34) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%cst_48 = arith.constant 0.22819308603366059 : f64
%cst_49 = arith.constant 0.32518251289747541 : f64
%87 = complex.create %cst_48, %cst_49 : complex<f64>
%c64_i32_50 = arith.constant 64 : i32
%c256_i32_51 = arith.constant 256 : i32
%c512_i32_52 = arith.constant 512 : i32
%88 = pulse.gaussian_square(%c512_i32_52, %87, %c64_i32_50, %c256_i32_51) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%86, %88) : (!pulse.port, !pulse.waveform)
%89 = "pulse.select_port"(%34) {id = "u1"} : (!pulse.port_group) -> !pulse.port
%cst_53 = arith.constant 1.5707963267948966 : f64
pulse.shift_phase(%89, %cst_53) : (!pulse.port, f64)
%90 = "pulse.select_port"(%41) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%cst_54 = arith.constant -1.5707963267948966 : f64
pulse.shift_phase(%90, %cst_54) : (!pulse.port, f64)
%91 = "pulse.select_port"(%41) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%cst_55 = arith.constant 0.089776566073944741 : f64
%cst_56 = arith.constant -3.3436108695185151E-4 : f64
%92 = complex.create %cst_55, %cst_56 : complex<f64>
%c40_i32_57 = arith.constant 40 : i32
%cst_58 = arith.constant 0.74011253340348393 : f64
%c160_i32_59 = arith.constant 160 : i32
%93 = pulse.drag(%c160_i32_59, %92, %c40_i32_57, %cst_58) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%91, %93) : (!pulse.port, !pulse.waveform)
%94 = "pulse.select_port"(%41) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%c672_i32_60 = arith.constant 672 : i32
pulse.delay(%c672_i32_60, %94) : (i32, !pulse.port)
%95 = "pulse.select_port"(%41) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%cst_61 = arith.constant 0.17980737117787168 : f64
%cst_62 = arith.constant 0.000000e+00 : f64
%96 = complex.create %cst_61, %cst_62 : complex<f64>
%c40_i32_63 = arith.constant 40 : i32
%cst_64 = arith.constant 0.6999359173856885 : f64
%c160_i32_65 = arith.constant 160 : i32
%97 = pulse.drag(%c160_i32_65, %96, %c40_i32_63, %cst_64) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%95, %97) : (!pulse.port, !pulse.waveform)
%98 = "pulse.select_port"(%41) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%c1344_i32 = arith.constant 1344 : i32
pulse.delay(%c1344_i32, %98) : (i32, !pulse.port)
%99 = "pulse.select_port"(%41) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%cst_66 = arith.constant -3.3436108695188198E-4 : f64
%cst_67 = arith.constant -0.089776566073944741 : f64
%100 = complex.create %cst_66, %cst_67 : complex<f64>
%c40_i32_68 = arith.constant 40 : i32
%cst_69 = arith.constant 0.74011253340348393 : f64
%c160_i32_70 = arith.constant 160 : i32
%101 = pulse.drag(%c160_i32_70, %100, %c40_i32_68, %cst_69) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%99, %101) : (!pulse.port, !pulse.waveform)
%102 = "pulse.select_port"(%41) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_71 = arith.constant -3.1415926535897931 : f64
pulse.shift_phase(%102, %cst_71) : (!pulse.port, f64)
%103 = "pulse.select_port"(%41) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_72 = arith.constant -5.1781053983669201E-4 : f64
%cst_73 = arith.constant 0.084483000576426853 : f64
%104 = complex.create %cst_72, %cst_73 : complex<f64>
%c40_i32_74 = arith.constant 40 : i32
%cst_75 = arith.constant -0.63693584079030685 : f64
%c160_i32_76 = arith.constant 160 : i32
%105 = pulse.drag(%c160_i32_76, %104, %c40_i32_74, %cst_75) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%103, %105) : (!pulse.port, !pulse.waveform)
%106 = "pulse.select_port"(%41) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%c160_i32_77 = arith.constant 160 : i32
pulse.delay(%c160_i32_77, %106) : (i32, !pulse.port)
%107 = "pulse.select_port"(%41) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_78 = arith.constant 0.061110783182318495 : f64
%cst_79 = arith.constant -1.5531471574357976E-4 : f64
%108 = complex.create %cst_78, %cst_79 : complex<f64>
%c64_i32_80 = arith.constant 64 : i32
%c256_i32_81 = arith.constant 256 : i32
%c512_i32_82 = arith.constant 512 : i32
%109 = pulse.gaussian_square(%c512_i32_82, %108, %c64_i32_80, %c256_i32_81) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%107, %109) : (!pulse.port, !pulse.waveform)
%110 = "pulse.select_port"(%41) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%c832_i32_83 = arith.constant 832 : i32
pulse.delay(%c832_i32_83, %110) : (i32, !pulse.port)
%111 = "pulse.select_port"(%41) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_84 = arith.constant -0.061110783182318495 : f64
%cst_85 = arith.constant 1.5531471574358724E-4 : f64
%112 = complex.create %cst_84, %cst_85 : complex<f64>
%c64_i32_86 = arith.constant 64 : i32
%c256_i32_87 = arith.constant 256 : i32
%c512_i32_88 = arith.constant 512 : i32
%113 = pulse.gaussian_square(%c512_i32_88, %112, %c64_i32_86, %c256_i32_87) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%111, %113) : (!pulse.port, !pulse.waveform)
%114 = "pulse.select_port"(%41) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%c1344_i32_89 = arith.constant 1344 : i32
pulse.delay(%c1344_i32_89, %114) : (i32, !pulse.port)
%115 = "pulse.select_port"(%41) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_90 = arith.constant -1.5707963267948966 : f64
pulse.shift_phase(%115, %cst_90) : (!pulse.port, f64)
%116 = "pulse.select_port"(%41) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%c1344_i32_91 = arith.constant 1344 : i32
pulse.delay(%c1344_i32_91, %116) : (i32, !pulse.port)
%117 = "pulse.select_port"(%41) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_92 = arith.constant 0.084483000576426853 : f64
%cst_93 = arith.constant 5.1781053983668951E-4 : f64
%118 = complex.create %cst_92, %cst_93 : complex<f64>
%c40_i32_94 = arith.constant 40 : i32
%cst_95 = arith.constant -0.63693584079030685 : f64
%c160_i32_96 = arith.constant 160 : i32
%119 = pulse.drag(%c160_i32_96, %118, %c40_i32_94, %cst_95) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%117, %119) : (!pulse.port, !pulse.waveform)
%120 = "pulse.select_port"(%41) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%cst_97 = arith.constant -3.1415926535897931 : f64
pulse.shift_phase(%120, %cst_97) : (!pulse.port, f64)
%121 = "pulse.select_port"(%41) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%c160_i32_98 = arith.constant 160 : i32
pulse.delay(%c160_i32_98, %121) : (i32, !pulse.port)
%122 = "pulse.select_port"(%41) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%cst_99 = arith.constant -0.22819308603366056 : f64
%cst_100 = arith.constant -0.32518251289747546 : f64
%123 = complex.create %cst_99, %cst_100 : complex<f64>
%c64_i32_101 = arith.constant 64 : i32
%c256_i32_102 = arith.constant 256 : i32
%c512_i32_103 = arith.constant 512 : i32
%124 = pulse.gaussian_square(%c512_i32_103, %123, %c64_i32_101, %c256_i32_102) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%122, %124) : (!pulse.port, !pulse.waveform)
%125 = "pulse.select_port"(%41) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%c832_i32_104 = arith.constant 832 : i32
pulse.delay(%c832_i32_104, %125) : (i32, !pulse.port)
%126 = "pulse.select_port"(%41) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%cst_105 = arith.constant 0.22819308603366059 : f64
%cst_106 = arith.constant 0.32518251289747541 : f64
%127 = complex.create %cst_105, %cst_106 : complex<f64>
%c64_i32_107 = arith.constant 64 : i32
%c256_i32_108 = arith.constant 256 : i32
%c512_i32_109 = arith.constant 512 : i32
%128 = pulse.gaussian_square(%c512_i32_109, %127, %c64_i32_107, %c256_i32_108) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%126, %128) : (!pulse.port, !pulse.waveform)
%129 = "pulse.select_port"(%41) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%c1344_i32_110 = arith.constant 1344 : i32
pulse.delay(%c1344_i32_110, %129) : (i32, !pulse.port)
%130 = "pulse.select_port"(%41) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%cst_111 = arith.constant -1.5707963267948966 : f64
pulse.shift_phase(%130, %cst_111) : (!pulse.port, f64)
%131 = "pulse.select_port"(%41) {id = "u1"} : (!pulse.port_group) -> !pulse.port
%cst_112 = arith.constant -1.5707963267948966 : f64
pulse.shift_phase(%131, %cst_112) : (!pulse.port, f64)
%132 = "pulse.select_port"(%41) {id = "u4"} : (!pulse.port_group) -> !pulse.port
%cst_113 = arith.constant -3.1415926535897931 : f64
pulse.shift_phase(%132, %cst_113) : (!pulse.port, f64)
%133 = "pulse.select_port"(%41) {id = "u4"} : (!pulse.port_group) -> !pulse.port
%c1344_i32_114 = arith.constant 1344 : i32
pulse.delay(%c1344_i32_114, %133) : (i32, !pulse.port)
%134 = "pulse.select_port"(%41) {id = "u4"} : (!pulse.port_group) -> !pulse.port
%cst_115 = arith.constant -1.5707963267948966 : f64
pulse.shift_phase(%134, %cst_115) : (!pulse.port, f64)
%135 = "pulse.select_port"(%41) {id = "u8"} : (!pulse.port_group) -> !pulse.port
%cst_116 = arith.constant -3.1415926535897931 : f64
pulse.shift_phase(%135, %cst_116) : (!pulse.port, f64)
%136 = "pulse.select_port"(%41) {id = "u8"} : (!pulse.port_group) -> !pulse.port
%c1344_i32_117 = arith.constant 1344 : i32
pulse.delay(%c1344_i32_117, %136) : (i32, !pulse.port)
%137 = "pulse.select_port"(%41) {id = "u8"} : (!pulse.port_group) -> !pulse.port
%cst_118 = arith.constant -1.5707963267948966 : f64
pulse.shift_phase(%137, %cst_118) : (!pulse.port, f64)
%138 = "pulse.select_port"(%41) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_119 = arith.constant 1.5707963267948966 : f64
pulse.shift_phase(%138, %cst_119) : (!pulse.port, f64)
%139 = "pulse.select_port"(%41) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_120 = arith.constant -3.1130851755889356E-17 : f64
%cst_121 = arith.constant -0.16946846376475716 : f64
%140 = complex.create %cst_120, %cst_121 : complex<f64>
%c40_i32_122 = arith.constant 40 : i32
%cst_123 = arith.constant -0.68671951029591838 : f64
%c160_i32_124 = arith.constant 160 : i32
%141 = pulse.drag(%c160_i32_124, %140, %c40_i32_122, %cst_123) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%139, %141) : (!pulse.port, !pulse.waveform)
%142 = "pulse.select_port"(%41) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%c912_i32 = arith.constant 912 : i32
pulse.delay(%c912_i32, %142) : (i32, !pulse.port)
%143 = "pulse.select_port"(%41) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_125 = arith.constant 0.16946846376475716 : f64
%cst_126 = arith.constant 0.000000e+00 : f64
%144 = complex.create %cst_125, %cst_126 : complex<f64>
%c40_i32_127 = arith.constant 40 : i32
%cst_128 = arith.constant -0.68671951029591838 : f64
%c160_i32_129 = arith.constant 160 : i32
%145 = pulse.drag(%c160_i32_129, %144, %c40_i32_127, %cst_128) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%143, %145) : (!pulse.port, !pulse.waveform)
%146 = "pulse.select_port"(%41) {id = "d2"} : (!pulse.port_group) -> !pulse.port
%cst_130 = arith.constant 0.094540487800318596 : f64
%cst_131 = arith.constant -3.9710888891587696E-4 : f64
%147 = complex.create %cst_130, %cst_131 : complex<f64>
%c40_i32_132 = arith.constant 40 : i32
%cst_133 = arith.constant 0.85354566868785298 : f64
%c160_i32_134 = arith.constant 160 : i32
%148 = pulse.drag(%c160_i32_134, %147, %c40_i32_132, %cst_133) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%146, %148) : (!pulse.port, !pulse.waveform)
%149 = "pulse.select_port"(%41) {id = "d2"} : (!pulse.port_group) -> !pulse.port
%c160_i32_135 = arith.constant 160 : i32
pulse.delay(%c160_i32_135, %149) : (i32, !pulse.port)
%150 = "pulse.select_port"(%41) {id = "d2"} : (!pulse.port_group) -> !pulse.port
%cst_136 = arith.constant 0.039131757733167315 : f64
%cst_137 = arith.constant -6.663466510650446E-4 : f64
%151 = complex.create %cst_136, %cst_137 : complex<f64>
%c64_i32_138 = arith.constant 64 : i32
%c496_i32 = arith.constant 496 : i32
%c752_i32 = arith.constant 752 : i32
%152 = pulse.gaussian_square(%c752_i32, %151, %c64_i32_138, %c496_i32) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%150, %152) : (!pulse.port, !pulse.waveform)
%153 = "pulse.select_port"(%41) {id = "d2"} : (!pulse.port_group) -> !pulse.port
%c1072_i32 = arith.constant 1072 : i32
pulse.delay(%c1072_i32, %153) : (i32, !pulse.port)
%154 = "pulse.select_port"(%41) {id = "d2"} : (!pulse.port_group) -> !pulse.port
%cst_139 = arith.constant -0.039131757733167315 : f64
%cst_140 = arith.constant 6.6634665106504937E-4 : f64
%155 = complex.create %cst_139, %cst_140 : complex<f64>
%c64_i32_141 = arith.constant 64 : i32
%c496_i32_142 = arith.constant 496 : i32
%c752_i32_143 = arith.constant 752 : i32
%156 = pulse.gaussian_square(%c752_i32_143, %155, %c64_i32_141, %c496_i32_142) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%154, %156) : (!pulse.port, !pulse.waveform)
%157 = "pulse.select_port"(%41) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%cst_144 = arith.constant 1.5707963267948966 : f64
pulse.shift_phase(%157, %cst_144) : (!pulse.port, f64)
%158 = "pulse.select_port"(%41) {id = "u2"} : (!pulse.port_group) -> !pulse.port
%c160_i32_145 = arith.constant 160 : i32
pulse.delay(%c160_i32_145, %158) : (i32, !pulse.port)
%159 = "pulse.select_port"(%41) {id = "u2"} : (!pulse.port_group) -> !pulse.port
%cst_146 = arith.constant -0.4672295345405707 : f64
%cst_147 = arith.constant -0.010192806263952648 : f64
%160 = complex.create %cst_146, %cst_147 : complex<f64>
%c64_i32_148 = arith.constant 64 : i32
%c496_i32_149 = arith.constant 496 : i32
%c752_i32_150 = arith.constant 752 : i32
%161 = pulse.gaussian_square(%c752_i32_150, %160, %c64_i32_148, %c496_i32_149) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%159, %161) : (!pulse.port, !pulse.waveform)
%162 = "pulse.select_port"(%41) {id = "u2"} : (!pulse.port_group) -> !pulse.port
%c1072_i32_151 = arith.constant 1072 : i32
pulse.delay(%c1072_i32_151, %162) : (i32, !pulse.port)
%163 = "pulse.select_port"(%41) {id = "u2"} : (!pulse.port_group) -> !pulse.port
%cst_152 = arith.constant 0.4672295345405707 : f64
%cst_153 = arith.constant 0.010192806263952591 : f64
%164 = complex.create %cst_152, %cst_153 : complex<f64>
%c64_i32_154 = arith.constant 64 : i32
%c496_i32_155 = arith.constant 496 : i32
%c752_i32_156 = arith.constant 752 : i32
%165 = pulse.gaussian_square(%c752_i32_156, %164, %c64_i32_154, %c496_i32_155) : (i32, complex<f64>, i32, i32) -> !pulse.waveform
pulse.play(%163, %165) : (!pulse.port, !pulse.waveform)
%166 = "pulse.select_port"(%41) {id = "u4"} : (!pulse.port_group) -> !pulse.port
%cst_157 = arith.constant 1.5707963267948966 : f64
pulse.shift_phase(%166, %cst_157) : (!pulse.port, f64)
%167 = "pulse.select_port"(%41) {id = "u8"} : (!pulse.port_group) -> !pulse.port
%cst_158 = arith.constant 1.5707963267948966 : f64
pulse.shift_phase(%167, %cst_158) : (!pulse.port, f64)
%168 = quir.constant #quir.angle<3.140000e+00 : !quir.angle<10>>
%169 = quir.constant #quir.angle<1.070000e+00 : !quir.angle<10>>
%170 = quir.constant #quir.angle<5.350000e-01 : !quir.angle<10>>
%171 = "pulse.select_port"(%34) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%cst_159 = arith.constant 0.000000e+00 : f64
pulse.shift_phase(%171, %cst_159) : (!pulse.port, f64)
%172 = "pulse.select_port"(%34) {id = "u1"} : (!pulse.port_group) -> !pulse.port
%cst_160 = arith.constant 0.000000e+00 : f64
pulse.shift_phase(%172, %cst_160) : (!pulse.port, f64)
%173 = "pulse.select_port"(%41) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_161 = arith.constant 0.000000e+00 : f64
pulse.shift_phase(%173, %cst_161) : (!pulse.port, f64)
%174 = "pulse.select_port"(%41) {id = "u0"} : (!pulse.port_group) -> !pulse.port
%cst_162 = arith.constant 0.000000e+00 : f64
pulse.shift_phase(%174, %cst_162) : (!pulse.port, f64)
%175 = "pulse.select_port"(%41) {id = "u4"} : (!pulse.port_group) -> !pulse.port
%cst_163 = arith.constant 0.000000e+00 : f64
pulse.shift_phase(%175, %cst_163) : (!pulse.port, f64)
%176 = "pulse.select_port"(%41) {id = "u8"} : (!pulse.port_group) -> !pulse.port
%cst_164 = arith.constant 0.000000e+00 : f64
pulse.shift_phase(%176, %cst_164) : (!pulse.port, f64)
%177 = "pulse.select_port"(%34) {id = "d0"} : (!pulse.port_group) -> !pulse.port
%cst_165 = arith.constant 0.089776566073944741 : f64
%cst_166 = arith.constant -3.3436108695185151E-4 : f64
%178 = complex.create %cst_165, %cst_166 : complex<f64>
%c40_i32_167 = arith.constant 40 : i32
%cst_168 = arith.constant 0.74011253340348393 : f64
%c160_i32_169 = arith.constant 160 : i32
%179 = pulse.drag(%c160_i32_169, %178, %c40_i32_167, %cst_168) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%177, %179) : (!pulse.port, !pulse.waveform)
%180 = "pulse.select_port"(%41) {id = "d1"} : (!pulse.port_group) -> !pulse.port
%cst_170 = arith.constant 0.084483000576426853 : f64
%cst_171 = arith.constant 5.1781053983668951E-4 : f64
%181 = complex.create %cst_170, %cst_171 : complex<f64>
%c40_i32_172 = arith.constant 40 : i32
%cst_173 = arith.constant -0.63693584079030685 : f64
%c160_i32_174 = arith.constant 160 : i32
%182 = pulse.drag(%c160_i32_174, %181, %c40_i32_172, %cst_173) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%180, %182) : (!pulse.port, !pulse.waveform)
%183 = "pulse.select_port"(%45) {id = "d2"} : (!pulse.port_group) -> !pulse.port
%cst_175 = arith.constant 0.094540487800318596 : f64
%cst_176 = arith.constant -3.9710888891587696E-4 : f64
%184 = complex.create %cst_175, %cst_176 : complex<f64>
%c40_i32_177 = arith.constant 40 : i32
%cst_178 = arith.constant 0.85354566868785298 : f64
%c160_i32_179 = arith.constant 160 : i32
%185 = pulse.drag(%c160_i32_179, %184, %c40_i32_177, %cst_178) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%183, %185) : (!pulse.port, !pulse.waveform)
%186 = "pulse.select_port"(%49) {id = "d3"} : (!pulse.port_group) -> !pulse.port
%cst_180 = arith.constant 0.095293610036729287 : f64
%cst_181 = arith.constant 0.0010061143295647052 : f64
%187 = complex.create %cst_180, %cst_181 : complex<f64>
%c40_i32_182 = arith.constant 40 : i32
%cst_183 = arith.constant 0.74995439781122553 : f64
%c160_i32_184 = arith.constant 160 : i32
%188 = pulse.drag(%c160_i32_184, %187, %c40_i32_182, %cst_183) : (i32, complex<f64>, i32, f64) -> !pulse.waveform
pulse.play(%186, %188) : (!pulse.port, !pulse.waveform)
%c0_i32 = arith.constant 0 : i32
return %c0_i32 : i32
}
