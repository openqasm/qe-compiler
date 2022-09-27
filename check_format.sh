#!/bin/bash

# Expects first argument as path to clang-format binary
# following arguments as paths to files to format

clang_format_bin=$1
files="${@:2}"

diff <(cat ${files}) <(${clang_format_bin} --style=file ${files})
