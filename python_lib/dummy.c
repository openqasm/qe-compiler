// This distribution contains platform-specific C++ libraries, but they are not
// built with distutils. We create a dummy Extension object so that distutils
// knows to make the binary platform-specific.

#include <stdio.h>

int main() { return 0; }

void initdummy() {}
