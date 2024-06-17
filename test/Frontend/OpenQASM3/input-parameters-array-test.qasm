OPENQASM 3;
// RUN: qss-compiler -X=qasm --enable-parameters --enable-circuits-from-qasm %s | FileCheck %s

gate rz(theta) q {}

qubit $0;

input angle angle_p;
input float[64] float_p;

input array[angle, 10] array_angle_p;
input array[float[64], 10] array_float_p;

// Access value directly
rz(angle_p) $0;
rz(float_p) $0;

// Access value through array reference
rz(array_angle_p[0]) $0;
rz(array_angle_p[1]) $0;
rz(array_float_p[0]) $0;
rz(array_float_p[1]) $0;

// Access value through intermediate assignment
float[64] intermediate_angle = array_angle_p[0];
float[64] intermediate_float = array_float_p[0];
rz(intermediate_angle) $0;
rz(intermediate_float) $0;

bit final = measure $0;
