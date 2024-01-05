// RUN: qss-compiler --target Mock --config path/to/config --allow-unregistered-dialect=false \
// RUN:          --add-target-passes=false --verbosity=info --show-config | FileCheck %s --check-prefix CLI
// RUN: QSSC_TARGET_NAME="MockEnv" QSSC_TARGET_CONFIG_PATH="path/to/config/Env" QSSC_VERBOSITY=DEBUG \
// RUN:          qss-compiler --allow-unregistered-dialect=false --add-target-passes=false --show-config | FileCheck %s --check-prefix ENV

// CLI: inputSource: -
// CLI: directInput: 0
// CLI: outputFilePath: -
// CLI: inputType: none
// CLI: emitAction: mlir
// CLI: targetName: Mock
// CLI: targetConfigPath: path/to/config
// CLI: verbosity: Info
// CLI: addTargetPasses: 0
// CLI: showTargets: 0
// CLI: showPayloads: 0
// CLI: showConfig: 1
// CLI: emitPlaintextPayload: 0
// CLI: includeSource: 0
// CLI: compileTargetIR: 0
// CLI: bypassPayloadTargetCompilation: 0

// CLI: allowUnregisteredDialects: 0
// CLI: dumpPassPipeline: 0
// CLI: emitBytecode: 0
// CLI: bytecodeEmitVersion: None
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
// ENV: allowUnregisteredDialects: 0
