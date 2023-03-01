//===- api.cpp --------------------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2023.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//

#include "API/api.h"

#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"

#include "llvm/ADT/Optional.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "Payload/Payload.h"
#include "QSSC.h"

#include "HAL/PassRegistration.h"
#include "HAL/TargetSystemRegistry.h"
#include "HAL/TargetSystem.h"

#include "Dialect/RegisterDialects.h"

#include "Dialect/Pulse/IR/PulseDialect.h"
#include "Dialect/Pulse/Transforms/Passes.h"

#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/Transforms/Passes.h"

#include "Dialect/QCS/IR/QCSDialect.h"

#include "Frontend/OpenQASM3/OpenQASM3Frontend.h"

#include <filesystem>
#include <utility>

using namespace mlir;

// The space below at the front of the string causes this category to be printed
// first
llvm::cl::OptionCategory
    qsscCat(" QSS Compiler Options",
            "Options that control high-level behavior of QSS Compiler");

static llvm::cl::opt<std::string>
    inputSource(llvm::cl::Positional,
                llvm::cl::desc("Input filename or program source"),
                llvm::cl::init("-"), llvm::cl::cat(qsscCat));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"),
                   llvm::cl::cat(qsscCat));

static llvm::cl::opt<std::string> configurationPath(
    "config",
    llvm::cl::desc("Path to configuration file or directory (depends on the "
                   "target), - means use the config service"),
    llvm::cl::value_desc("path"), llvm::cl::cat(qsscCat));

static llvm::cl::opt<std::string>
    targetStr("target",
              llvm::cl::desc(
                  "Target architecture. Required for machine code generation."),
              llvm::cl::value_desc("targetName"), llvm::cl::cat(qsscCat));

static llvm::cl::opt<bool>
    directInput("direct",
                llvm::cl::desc("Accept the input program directly as a string"),
                llvm::cl::cat(qsscCat));

static llvm::cl::opt<bool>
    addTargetPasses("add-target-passes",
                    llvm::cl::desc("Add target-specific passes"),
                    llvm::cl::init(true), llvm::cl::cat(qsscCat));

/*
// This is mainly used for increasing test speed
// not used currently, may enable in future
static llvm::cl::opt<bool> splitInputFile(
    "split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently"),
    llvm::cl::init(false));
*/

static llvm::cl::opt<bool> verifyDiagnostics(
    "verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::init(false), llvm::cl::cat(qsscCat));

static llvm::cl::opt<bool> verifyPasses(
    "verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::init(true), llvm::cl::cat(qsscCat));

static llvm::cl::opt<bool> allowUnregisteredDialects(
    "allow-unregistered-dialect",
    llvm::cl::desc("Allow operation with no registered dialects"),
    llvm::cl::init(false), llvm::cl::cat(qsscCat));

static llvm::cl::opt<bool>
    showDialects("show-dialects",
                 llvm::cl::desc("Print the list of registered dialects"),
                 llvm::cl::init(false), llvm::cl::cat(qsscCat));

static llvm::cl::opt<bool>
    showTargets("show-targets",
                llvm::cl::desc("Print the list of registered targets"),
                llvm::cl::init(false), llvm::cl::cat(qsscCat));

static llvm::cl::opt<bool>
    plaintextPayload("plaintext-payload",
                     llvm::cl::desc("Write the payload in plaintext"),
                     llvm::cl::init(false), llvm::cl::cat(qsscCat));

namespace {
enum InputType { NONE, QASM, MLIR, QOBJ };
} // anonymous namespace
static llvm::cl::opt<enum InputType> inputType(
    "X", llvm::cl::init(InputType::NONE),
    llvm::cl::desc("Specify the kind of input desired"),
    llvm::cl::values(
        clEnumValN(InputType::QASM, "qasm",
                   "load the input file as an OpenQASM 3.0 source")),
    llvm::cl::values(clEnumValN(MLIR, "mlir",
                                "load the input file as an MLIR file")),
    llvm::cl::values(clEnumValN(QOBJ, "qobj",
                                "load the input file as a QOBJ file")));

namespace {
enum Action { None, DumpAST, DumpASTPretty, DumpMLIR, DumpWaveMem, GenQEM };
} // anonymous namespace
static llvm::cl::opt<enum Action> emitAction(
    "emit", llvm::cl::init(Action::None),
    llvm::cl::desc("Select the kind of output desired"),
    llvm::cl::values(clEnumValN(DumpAST, "ast", "output the AST dump")),
    llvm::cl::values(clEnumValN(DumpASTPretty, "ast-pretty",
                                "pretty print the AST")),
    llvm::cl::values(clEnumValN(DumpMLIR, "mlir", "output the MLIR dump")),
    llvm::cl::values(clEnumValN(DumpWaveMem, "wavemem",
                                "output the waveform memory")),
    llvm::cl::values(clEnumValN(GenQEM, "qem",
                                "generate a quantum executable module (qem) "
                                "for execution on hardware")));

namespace qss {
enum FileExtension { None, AST, ASTPRETTY, QASM, QOBJ, MLIR, WMEM, QEM };
} // namespace qss

auto fileExtensionToStr(const qss::FileExtension &inExt) -> std::string {
  switch (inExt) {
  case qss::FileExtension::AST:
    return "ast";
    break;
  case qss::FileExtension::ASTPRETTY:
    return "ast-pretty";
    break;
  case qss::FileExtension::QASM:
    return "qasm";
    break;
  case qss::FileExtension::QOBJ:
    return "qobj";
    break;
  case qss::FileExtension::MLIR:
    return "mlir";
    break;
  case qss::FileExtension::WMEM:
    return "wmem";
    break;
  case qss::FileExtension::QEM:
    return "qem";
    break;
  default:
    return "none";
    break;
  }
  return "none";
}

auto fileExtensionToInputType(const qss::FileExtension &inExt) -> InputType {
  switch (inExt) {
  case qss::FileExtension::QASM:
    return InputType::QASM;
    break;
  case qss::FileExtension::QOBJ:
    return InputType::QOBJ;
    break;
  case qss::FileExtension::MLIR:
    return InputType::MLIR;
    break;
  default:
    break;
  }
  return InputType::NONE;
}

auto fileExtensionToAction(const qss::FileExtension &inExt) -> Action {
  switch (inExt) {
  case qss::FileExtension::AST:
    return Action::DumpAST;
    break;
  case qss::FileExtension::ASTPRETTY:
    return Action::DumpASTPretty;
    break;
  case qss::FileExtension::MLIR:
    return Action::DumpMLIR;
    break;
  case qss::FileExtension::WMEM:
    return Action::DumpWaveMem;
    break;
  case qss::FileExtension::QEM:
    return Action::GenQEM;
    break;
  default:
    break;
  }
  return Action::None;
}

auto strToFileExtension(const std::string &extStr) -> qss::FileExtension {
  if (extStr == "ast" || extStr == "AST")
    return qss::FileExtension::AST;
  if (extStr == "ast-pretty" || extStr == "AST-PRETTY")
    return qss::FileExtension::ASTPRETTY;
  if (extStr == "qasm" || extStr == "QASM")
    return qss::FileExtension::QASM;
  if (extStr == "qobj" || extStr == "QOBJ")
    return qss::FileExtension::QOBJ;
  if (extStr == "mlir" || extStr == "MLIR")
    return qss::FileExtension::MLIR;
  if (extStr == "wmem" || extStr == "WMEM")
    return qss::FileExtension::WMEM;
  if (extStr == "qem" || extStr == "QEM")
    return qss::FileExtension::QEM;
  return qss::FileExtension::None;
}

// extracts the file extension and returns the enum qss::FileExtension type
auto getExtension(const std::string &inStr) -> qss::FileExtension {
  auto pos = inStr.find_last_of('.');
  if (pos < inStr.length())
    return strToFileExtension(inStr.substr(pos + 1));
  return qss::FileExtension::None;
}

llvm::Error registerPasses() {
  // TODO: Register standalone passes here.
  llvm::Error err = llvm::Error::success();
  mlir::quir::registerQuirPasses();
  mlir::quir::registerQuirPassPipeline();
  mlir::pulse::registerPulsePasses();
  mlir::pulse::registerPulsePassPipeline();
  mlir::registerConversionPasses();

  err = llvm::joinErrors(std::move(err), qssc::hal::registerTargetPasses());
  err = llvm::joinErrors(std::move(err), qssc::hal::registerTargetPipelines());

  mlir::registerAllPasses();
  return err;
}

auto registerPassManagerCLOpts() {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();
  mlir::registerPassManagerCLOptions();
  mlir::registerDefaultTimingManagerCLOptions();
}

llvm::Error determineInputType() {
  if (inputType == InputType::NONE) {
    inputType = fileExtensionToInputType(getExtension(inputSource));
    if (inputType == InputType::NONE) {
      if (directInput) {
        inputType = InputType::QASM;
      } else {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "Unable to autodetect file extension type! Please specify the "
            "input type with -X");
      }
    } else if (directInput && inputType != InputType::QASM) {
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "Can only compile direct input when the input type is QASM");
    }
  }
  return llvm::Error::success();
}

void determineOutputType() {
  if (outputFilename != "-") {
    Action extensionAction =
        fileExtensionToAction(getExtension(outputFilename));
    if (extensionAction == Action::None && emitAction == Action::None) {
      llvm::errs()
          << "Can't figure out file extension of specified output file "
          << outputFilename << " defaulting to dumping MLIR\n";
      emitAction = Action::DumpMLIR;
    } else if (emitAction == Action::None) {
      emitAction = extensionAction;
    } else if (extensionAction != emitAction) {
      llvm::errs() << "Warning! The output type in the file extension doesn't "
                      "match the output type specified by --emit!\n";
    }
  } else {
    if (emitAction == Action::None)
      emitAction = Action::DumpMLIR;
  }
}

static void printVersion(llvm::raw_ostream &out) {
  out << "Quantum System Software (QSS) compiler version "
      << qssc::getQSSCVersion() << "\n";
}

static llvm::Error
compile_(int argc, char const **argv, std::string *outputString,
         llvm::Optional<qssc::DiagnosticCallback> diagnosticCb) {
  llvm::InitLLVM y(argc, argv);

  if (auto err = registerPasses())
    return err;

  DialectRegistry registry = qssc::dialect::registerDialects();
  mlir::PassPipelineCLParser passPipeline("", "Compiler passes to run");
  registerPassManagerCLOpts();

  llvm::cl::SetVersionPrinter(&printVersion);
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "Quantum System Software (QSS) Backend Compiler\n");

  if (showDialects) {
    llvm::outs() << "Registered Dialects:\n";
    for (const auto &registeredDialect : registry.getDialectNames())
      llvm::outs() << registeredDialect << "\n";
    return llvm::Error::success();
  }

  if (showTargets) {
    llvm::outs() << "Registered Targets:\n";
    for (const auto &target : qssc::hal::registry::TargetSystemRegistry::registeredPlugins()) {
      // Constants chosen empirically to align with --help.
      // TODO: Select constants more intelligently.
      qssc::plugin::registry::printHelpStr(target.second, 2, 57);
    }
    return llvm::Error::success();
  }

  if (auto err = determineInputType())
    return err;

  determineOutputType();

  // Make sure target exists if specified
  if (!targetStr.empty() && !qssc::hal::registry::TargetSystemRegistry::pluginExists(targetStr))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Error: Target " + targetStr +
                                       " is not registered.");

  qssc::hal::registry::TargetSystemInfo &targetInfo =
      *qssc::hal::registry::TargetSystemRegistry::lookupPluginInfo(targetStr).getValueOr(
          qssc::hal::registry::TargetSystemRegistry::nullTargetSystemInfo());

  MLIRContext context{};
  llvm::Optional<llvm::StringRef> conf{};
  if (!configurationPath.empty())
    conf.emplace(configurationPath);
  auto created = targetInfo.createTarget(&context, conf);
  if (auto err = created.takeError()) {
    return llvm::joinErrors(
        llvm::createStringError(llvm::inconvertibleErrorCode(),
                                "Unable to create target!"),
        std::move(err));
  }

  qssc::hal::TargetSystem &target = *created.get();

  // Set up the input, which is loaded from a file by name by default. With the
  // "--direct" option, the input program can be provided as a string to stdin.
  std::string errorMessage;
  std::unique_ptr<llvm::MemoryBuffer> file;
  if (!directInput) {
    file = mlir::openInputFile(inputSource, &errorMessage);
    if (!file) {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Failed to open input file: " +
                                         errorMessage);
    }
  }

  // Set up the output.
  llvm::raw_ostream *ostream;
  llvm::Optional<llvm::raw_string_ostream> outStringStream;
  auto outputFile = mlir::openOutputFile(outputFilename, &errorMessage);
  std::unique_ptr<qssc::payload::Payload> payload = nullptr;

  if (emitAction == Action::GenQEM) {
    if (outputFilename == "-") {
      payload = std::make_unique<qssc::payload::ZipPayload>();
    } else {
      std::filesystem::path payloadPath(outputFilename.c_str());
      std::string fNamePrefix = payloadPath.stem();
      payload =
          std::make_unique<qssc::payload::ZipPayload>(fNamePrefix, fNamePrefix);
    }
  }
  if (outputString) {
    outStringStream.emplace(*outputString);
    ostream = outStringStream.getPointer();
  } else {
    if (!outputFile)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Failed to open output file: " +
                                         errorMessage);
    ostream = &outputFile->os();
  }

  mlir::ModuleOp moduleOp;

  context.appendDialectRegistry(registry);
  context.allowUnregisteredDialects(allowUnregisteredDialects);
  context.printOpOnDiagnostic(!verifyDiagnostics);

  if (inputType == InputType::QASM) {
    if (emitAction >= Action::DumpMLIR) {
      moduleOp = mlir::ModuleOp::create(FileLineColLoc::get(
          &context, directInput ? std::string{"-"} : inputSource, 0, 0));
    }

    if (auto frontendError = qssc::frontend::openqasm3::parse(
            inputSource, !directInput, emitAction == Action::DumpAST,
            emitAction == Action::DumpASTPretty, emitAction >= Action::DumpMLIR,
            moduleOp, std::move(diagnosticCb)))
      return frontendError;

    if (emitAction < Action::DumpMLIR)
      return llvm::Error::success();
  } // if input == QASM

  if (inputType == InputType::MLIR) {
    // ------------------------------------------------------------
    // The following section was copied from processBuffer() in:
    //      ../third_party/llvm-project/mlir/lib/Support/MlirOptMain.cpp

    // Tell sourceMgr about this buffer, which is what the parser will pick up.
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

    // Parse the input file.
    // ------------------------------------------------------------
    // The following section was copied from performActions() in:
    //      ../third_party/llvm-project/mlir/lib/Support/MlirOptMain.cpp

    // Disable multi-threading when parsing the input file. This removes the
    // unnecessary/costly context synchronization when parsing.
    bool wasThreadingEnabled = context.isMultithreadingEnabled();
    context.disableMultithreading();

    // Parse the input file and reset the context threading state.
    mlir::OwningOpRef<ModuleOp> module(
        mlir::parseSourceFile(sourceMgr, &context));
    context.enableMultithreading(wasThreadingEnabled);
    if (!module)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Problem parsing source file " +
                                         inputSource);
    moduleOp = module.release();
  } // if input == MLIR

  // at this point we have QUIR+Pulse in the moduleOp from either the
  // QASM/AST or MLIR file

  // Apply any pass manager command line options.
  mlir::PassManager pm(&context);
  mlir::applyPassManagerCLOptions(pm);
  mlir::applyDefaultTimingPassManagerCLOptions(pm);

  auto errorHandler = [&](const Twine &msg) {
    emitError(UnknownLoc::get(&context)) << msg;
    return failure();
  };

  // Build the provided pipeline.
  if (failed(passPipeline.addToPipeline(pm, errorHandler)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Problem adding passes to passPipeline!");

  if (emitAction > Action::DumpMLIR)
    // check if the target quir to std pass has been specified in the CL
    if (addTargetPasses)
      if (auto err = target.addPayloadPasses(pm))
        return llvm::joinErrors(
            llvm::createStringError(llvm::inconvertibleErrorCode(),
                                    "Failure while preparing target passes"),
            std::move(err));

  // Run the pipeline.
  if (failed(pm.run(moduleOp)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Problems running the compiler pipeline!");

  if (emitAction == Action::DumpMLIR) {
    // Print the output.
    moduleOp.print(*ostream);
    *ostream << '\n';
  }

  if (emitAction == Action::GenQEM) {
    if (auto err = target.addToPayload(moduleOp, *payload))
      return err;

    if (plaintextPayload)
      payload->writePlain(*ostream);
    else
      payload->write(*ostream);
  }

  // ------------------------------------------------------------

  // Keep the output if no errors have occurred so far
  if (outputString) {
    outStringStream.getValue().str();
    if (outputFile && outputFilename != "-")
      outputFile->os() << *outputString;
  }
  if (outputFile && outputFilename != "-")
    outputFile->keep();

  return llvm::Error::success();
}

int qssc::compile(int argc, char const **argv, std::string *outputString,
                  llvm::Optional<DiagnosticCallback> diagnosticCb) {
  if (auto err = compile_(argc, argv, outputString, std::move(diagnosticCb))) {
    llvm::logAllUnhandledErrors(std::move(err), llvm::errs(), "Error: ");
    return 1;
  }

  return 0;
}

llvm::Error qssc::bindParameters(
    llvm::StringRef target, llvm::StringRef moduleInputPath,
    llvm::StringRef payloadOutputPath,
    std::unordered_map<std::string, double> const &parameters) {

  // ZipPayloads are implemented with libzip, which only supports updating a zip
  // archive in-place. Thus, copy module to payload first, then update payload
  // (instead of read module, update, write payload)
  std::error_code copyError =
      llvm::sys::fs::copy_file(moduleInputPath, payloadOutputPath);

  if (copyError)
    return llvm::make_error<llvm::StringError>(
        "Failed to copy circuit module to payload", copyError);

  // TODO actually update parameters, tbd in later commits.

  return llvm::Error::success();
}
