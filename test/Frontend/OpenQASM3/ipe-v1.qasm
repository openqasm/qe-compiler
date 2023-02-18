// Read calibrations and enforce full schedule

// Compiler pass inserts defcals and resolves timing:
// For the scheduling, let's say that y90p is 20 ns on 0, 30 ns on 1, and cr90p on (0,1) is 300 ns.
// Let's say phase is 0 ns and the entire measurement is 400 ns.

// Questions:

// - How do we deal with casting to different precisions? Here the phase defcal takes 20-bit angles, but
//   the gate definitions above use 32-bit angles. The same question comes up with the openpulse pi keyword I suppose.
// - How are we going to do reset in this example? Measure and do a bit flip? Do we insert this in a
//   previous pass, or do we define this behavior in a defcal?

OPENQASM 3.0;
// RUN: qss-compiler -X=qasm --emit=ast %s

// include "openpulse.inc";
// defcalgrammar "openpulse";

qubit $0;
qubit $1;

// defcal y90p $0 {
//     play drive($0), drag(...);
// }

// defcal y90p $1 {
//     play drive($1), drag(...);
// }

// defcal cr90p $0, $1 {
//     play flat_top_gaussian(...), drive($0), frame(drive($1));
// }

// defcal phase(angle[20]: theta) $q {
//     shift_phase drive($q), -theta;
// }

// defcal cr90m $0, $1 {
//     phase(-pi) $1;
//     cr90p $0, $1;
//     phase(pi) $1;
// }

// defcal x90p $q {
//     phase(pi) $q;
//     y90p $q;
//     phase(-pi) $q;
// }

// defcal xp $q {
//     x90p $q;
//     x90p $q;
// }

// defcal h $q {
//     phase(pi) $q;
//     y90p $q;
// }

// defcal CX $control $target {
//   phase(-pi/2) $control;
//   xp $control;
//   x90p $target;
//   barrier $control, $target;
//   cr90p $control, $target;
//   barrier $control, $target;
//   xp $control;
//   barrier $control, $target;
//   cr90m $control, $target;
// }

// defcal measure $0 -> bit {
//      complex[int[24]] iq;
//      bit state;
//      complex[int[12]] k0[1024] = [i0 + q0*j, i1 + q1*j, i2 + q2*j, ...];
//      play measure($0), flat_top_gaussian(...);
//      iq = capture acquire($0), 2048, kernel(k0);
//      return threshold(iq, 1234);
// }
gate h q {
    U(pi, 0, pi) q;
}
gate cx qa, qb { }
gate phase(theta) qa {
    U(theta, 0, 0) qa;
}

angle[3] c = 0;

reset $0;
reset $1;
h $1;
h $0;
stretch a;
stretch b;
delay[a] $0;
delay[b] $1;
cx $0, $1;
phase(1.8125*pi) $1;
cx $0, $1;
phase(0.1875*pi) $1;
phase(0.1875*pi) $0;
h $0;
measure $0 -> c[0];
c <<= 1;

reset $0;
h $0;
stretch d;
delay[d] $1;
cx $0, $1;
phase(1.625*pi) $1;  // mod 2*pi
cx $0, $1;
phase(0.375*pi) $1;
angle[32] temp_1 = 0.375*pi;
temp_1 -= c;  // cast and do arithmetic mod 2 pi
phase(temp_1) $0;
h $0;
measure $0 -> c[0];
c <<= 1;

reset $0;
h $0;
stretch f;
delay[f] $1;
cx $0, $1;
phase(1.25*pi) $1;  // mod 2*pi
cx $0, $1;
phase(0.75*pi) $1;
angle[32] temp_2 = 0.75*pi;
temp_2 -= c;  // cast and do arithmetic mod 2 pi
phase(temp_2) $0;
h $0;
measure $0 -> c[0];
c <<= 1;
stretch g;
delay[g] $1;
