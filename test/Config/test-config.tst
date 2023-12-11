// RUN: qss-compiler --target Mock --config path/to/config --allow-unregistered-dialect=false \
// RUN:          --add-target-passes=false --verbosity=info --show-config | FileCheck %s --check-prefix CLI
// RUN: QSSC_TARGET_NAME="MockEnv" QSSC_TARGET_CONFIG_PATH="path/to/config/Env" QSSC_VERBOSITY=DEBUG \
// RUN:          qss-compiler --allow-unregistered-dialect=false --add-target-passes=false --show-config | FileCheck %s --check-prefix ENV

// CLI: targetName: Mock
// CLI: targetConfigPath: path/to/config
// CLI: allowUnregisteredDialects: 0
// CLI: addTargetPasses: 0
// CLI: verbosity: Info

// ENV: targetName: MockEnv
// ENV: targetConfigPath: path/to/config/Env
// ENV: allowUnregisteredDialects: 0
// ENV: addTargetPasses: 0
// ENV: verbosity: Debug
