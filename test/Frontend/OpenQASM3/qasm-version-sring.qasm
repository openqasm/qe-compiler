OPENQASM 3;

// RUN: qss-compiler -X=qasm --emit=mlir %s

// Error parsing the OPENQASM line directive.
// OPENQASM3;
