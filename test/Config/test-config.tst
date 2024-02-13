// RUN: qss-compiler --target Mock --config path/to/config --allow-unregistered-dialect=false \
// RUN:          --add-target-passes=false --verbosity=info --max-threads=5 --show-config - | FileCheck %s --check-prefix CLI
// RUN: QSSC_TARGET_NAME="MockEnv" QSSC_TARGET_CONFIG_PATH="path/to/config/Env" QSSC_VERBOSITY=DEBUG QSSC_MAX_THREADS=10 \
// RUN:          qss-compiler --allow-unregistered-dialect=false --add-target-passes=false --show-config - | FileCheck %s --check-prefix ENV
// REQUIRES: !asserts

// CLI: inputType: none
// CLI: emitAction: mlir
// CLI: targetName: Mock
// CLI: targetConfigPath: path/to/config
// CLI: verbosity: Info
// CLI: addTargetPasses: 0
// CLI: showTargets: 0
// CLI: showPayloads: 0
// CLI: showConfig: 1
// CLI: payloadName: -
// CLI: emitPlaintextPayload: 0
// CLI: includeSource: 0
// CLI: compileTargetIR: 0
// CLI: bypassPayloadTargetCompilation: 0
// CLI: maxThreads: 5

// CLI: allowUnregisteredDialects: 0
// CLI: dumpPassPipeline: 0
// CLI: emitBytecode: 0
// CLI: emitBytecodeVersion: None
// CLI: irdlFile:
// CLI: runReproducer: 0
// CLI: showDialects: 0
// CLI: splitInputFile: 0
// CLI: useExplicitModule: 0
// CLI: verifyDiagnostics: 0
// CLI: verifyPasses: 0
// CLI: verifyRoundTrip: 0

// ENV: targetName: MockEnv
// ENV: targetConfigPath: path/to/config/Env
// ENV: verbosity: Debug
// ENV: addTargetPasses: 0
// ENV: maxThreads: 10
// ENV: allowUnregisteredDialects: 0
