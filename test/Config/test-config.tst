// RUN: qss-compiler --target Mock --config path/to/config --allow-unregistered-dialect=false --add-target-passes=false --show-config | FileCheck %s --check-prefix CLI

// CLI: targetName: Mock
// CLI: targetConfigPath: path/to/config
// CLI: allowUnregisteredDialects: 0
// CLI: addTargetPasses: 0
