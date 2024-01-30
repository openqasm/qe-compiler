// Test emitting nothing. Useful for testing and benchmark purposes
// RUN: qss-compiler %s --emit=none | if [[ $(ls -A | head -c1 | wc -c) -ne 0 ]]; then exit 0; fi

func.func @dummy() {
    return
}
