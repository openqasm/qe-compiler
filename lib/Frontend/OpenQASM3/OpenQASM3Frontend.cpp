//===- OpenQASM3Frontend.cpp ------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
//
// This code is part of Qiskit.
//
// This code is licensed under the Apache License, Version 2.0 with LLVM
// Exceptions. You may obtain a copy of this license in the LICENSE.txt
// file in the root directory of this source tree.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.
//
//===----------------------------------------------------------------------===//
///
///  This file contains the implementation of the OpenQASM 3 frontend.
///
//===----------------------------------------------------------------------===//

#include "Frontend/OpenQASM3/OpenQASM3Frontend.h"

#include "API/errors.h"
#include "Dialect/QUIR/IR/QUIRDialect.h"
#include "Dialect/QUIR/IR/QUIREnums.h"
#include "Frontend/OpenQASM3/PrintQASM3Visitor.h"
#include "Frontend/OpenQASM3/QUIRGenQASM3Visitor.h"

#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/Timing.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <qasm/AST/ASTRoot.h>
#include <qasm/AST/ASTStatement.h>
#include <qasm/AST/ASTStatementBuilder.h>
#include <qasm/Frontend/QasmDiagnosticEmitter.h>
#include <qasm/Frontend/QasmParser.h>
#include <qasm/QPP/QasmPP.h>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <utility>

namespace {

llvm::cl::OptionCategory openqasm3Cat(
    " OpenQASM 3 Frontend Options",
    "Options that control the OpenQASM 3 frontend of QSS Compiler");

llvm::cl::opt<uint>
    numShots("num-shots",
             llvm::cl::desc("The number of shots to execute on the quantum "
                            "circuit, default is 1000"),
             llvm::cl::init(1000), llvm::cl::cat(openqasm3Cat));

llvm::cl::opt<std::string> shotDelay(
    "shot-delay",
    llvm::cl::desc("Repetition delay between shots. Defaults to 1ms."),
    llvm::cl::init("1ms"), llvm::cl::cat(openqasm3Cat));

llvm::cl::list<std::string>
    includeDirs("I", llvm::cl::desc("Add <dir> to the include path"),
                llvm::cl::value_desc("dir"), llvm::cl::cat(openqasm3Cat));

qssc::DiagnosticCallback *diagnosticCallback_;
llvm::SourceMgr *sourceMgr_;

std::mutex qasmParserLock;

std::regex durationRe("^([0-9]*[.]?[0-9]+)([a-zA-Z]*)");

llvm::Expected<std::pair<double, mlir::quir::TimeUnits>>
parseDurationStr(const std::string &durationStr) {
  std::smatch m;
  std::regex_match(durationStr, m, durationRe);
  if (m.size() != 3)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        llvm::Twine("Unable to parse duration from ") + durationStr);

  double const parsedDuration = std::stod(m[1]);
  // Convert all units to lower case.
  auto unitStr = m[2].str();
  auto lowerUnitStr = llvm::StringRef(unitStr).lower();
  if (lowerUnitStr == "")
    // Empty case is SI
    lowerUnitStr = "s";

  if (auto parsedUnits = mlir::quir::symbolizeTimeUnits(lowerUnitStr))
    return std::make_pair(parsedDuration, parsedUnits.value());

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 llvm::Twine("Unknown duration unit ") +
                                     unitStr);
}

} // anonymous namespace

llvm::Error qssc::frontend::openqasm3::parse(
    llvm::SourceMgr &sourceMgr, bool emitRawAST, bool emitPrettyAST,
    bool emitMLIR, mlir::ModuleOp newModule,
    qssc::OptDiagnosticCallback diagnosticCallback, mlir::TimingScope &timing) {

  const llvm::MemoryBuffer *sourceBuffer =
      sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());

  mlir::TimingScope qasm3ParseTiming = timing.nest("parse-qasm3");

  // The QASM parser can only be called from a single thread.
  std::lock_guard<std::mutex> const qasmParserLockGuard(qasmParserLock);

  for (const auto &dirStr : includeDirs)
    QASM::QasmPreprocessor::Instance().AddIncludePath(dirStr);

  QASM::ASTParser parser;
  auto root = std::unique_ptr<QASM::ASTRoot>(nullptr);

  // Add a callback for diagnostics to the parser. Since the callback needs
  // access to diagnosticCallback to forward diagnostics, make it available in a
  // global variable.
  diagnosticCallback_ =
      diagnosticCallback.has_value() ? &diagnosticCallback.value() : nullptr;
  sourceMgr_ = &sourceMgr;
  QASM::QasmDiagnosticEmitter::SetHandler(
      [](const std::string &File, QASM::ASTLocation Loc, // NOLINT
         const std::string &Msg, QASM::QasmDiagnosticEmitter::DiagLevel DL) {
        std::string level = "unknown";
        qssc::Severity diagLevel = qssc::Severity::Error;
        llvm::SourceMgr::DiagKind sourceMgrDiagKind =
            llvm::SourceMgr::DiagKind::DK_Error;

        switch (DL) {
        case QASM::QasmDiagnosticEmitter::DiagLevel::Error:
          level = "Error";
          diagLevel = qssc::Severity::Error;
          sourceMgrDiagKind = llvm::SourceMgr::DiagKind::DK_Error;
          break;

        case QASM::QasmDiagnosticEmitter::DiagLevel::ICE:
          level = "ICE";
          diagLevel = qssc::Severity::Fatal;
          sourceMgrDiagKind = llvm::SourceMgr::DiagKind::DK_Error;
          break;

        case QASM::QasmDiagnosticEmitter::DiagLevel::Warning:
          level = "Warning";
          diagLevel = qssc::Severity::Warning;
          sourceMgrDiagKind = llvm::SourceMgr::DiagKind::DK_Warning;
          break;

        case QASM::QasmDiagnosticEmitter::DiagLevel::Info:
          level = "Info";
          diagLevel = qssc::Severity::Info;
          sourceMgrDiagKind = llvm::SourceMgr::DiagKind::DK_Remark;
          break;

        case QASM::QasmDiagnosticEmitter::DiagLevel::Status:
          level = "Status";
          diagLevel = qssc::Severity::Info;
          sourceMgrDiagKind = llvm::SourceMgr::DiagKind::DK_Note;
          break;
        }

        // Capture source context for including it in error messages
        assert(sourceMgr_);
        auto &sourceMgr = *sourceMgr_;
        auto loc = sourceMgr.FindLocForLineAndColumn(1, Loc.LineNo, Loc.ColNo);
        std::string sourceString;
        llvm::raw_string_ostream stringStream(sourceString);

        sourceMgr.PrintMessage(stringStream, loc, sourceMgrDiagKind, "");

        std::stringstream fileLoc;
        fileLoc << "File: " << File << ", Line: " << Loc.LineNo
                << ", Col: " << Loc.ColNo;

        llvm::errs() << level << " while parsing OpenQASM 3 input\n"
                     << fileLoc.str() << " " << Msg << "\n"
                     << sourceString << "\n";

        if (diagnosticCallback_) {
          qssc::Diagnostic const diag{
              diagLevel, qssc::ErrorCategory::OpenQASM3ParseFailure,
              fileLoc.str() + "\n" + Msg + "\n" + sourceString};
          (*diagnosticCallback_)(diag);
        }

        if (DL == QASM::QasmDiagnosticEmitter::DiagLevel::Error ||
            DL == QASM::QasmDiagnosticEmitter::DiagLevel::ICE) {
          // give up parsing after errors right away
          // TODO: update to recent qss-qasm to support continuing
          throw std::runtime_error("Failure parsing");
        }
      });

  try {
    auto sourceFile = sourceBuffer->getBufferIdentifier();

    // Handle stdin differently as the qasm parser does not seem to perform
    // includes on a raw string input. This limits includes to file inputs only
    // at the current time.
    if (!(sourceFile == "" || sourceFile == "<stdin>")) {
      QASM::QasmPreprocessor::Instance().SetTranslationUnit(sourceFile.str());
      root.reset(parser.ParseAST());
    } else
      root.reset(parser.ParseAST(sourceBuffer->getBuffer().str()));

  } catch (std::exception &e) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        llvm::Twine{"Exception while parsing OpenQASM 3 input: "} + e.what());
  }

  qasm3ParseTiming.stop();

  if (!root)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "Failed to parse OpenQASM 3 input");

  if (emitRawAST)
    root->print();

  if (emitPrettyAST) {
    auto *statementList = QASM::ASTStatementBuilder::Instance().List();
    qssc::frontend::openqasm3::PrintQASM3Visitor visitor(std::cout);

    visitor.setStatementList(statementList);
    visitor.walkAST();
  }

  if (emitMLIR) {
    mlir::TimingScope qasm3ToMlirTiming = timing.nest("convert-qasm3-to-mlir");

    auto *context = newModule.getContext();

    context->loadDialect<mlir::quir::QUIRDialect>();
    context->loadDialect<mlir::complex::ComplexDialect>();
    context->loadDialect<mlir::func::FuncDialect>();

    mlir::OpBuilder const builder(newModule.getBodyRegion());

    QASM::ASTStatementList *statementList =
        QASM::ASTStatementBuilder::Instance().List();

    qssc::frontend::openqasm3::QUIRGenQASM3Visitor visitor(builder, newModule,
                                                           /*filename=*/"");

    auto result = parseDurationStr(shotDelay);
    if (auto err = result.takeError())
      return err;

    const auto [shotDelayValue, shotDelayUnits] = *result;
    visitor.initialize(numShots, shotDelayValue, shotDelayUnits);
    visitor.setStatementList(statementList);
    visitor.setInputFile(sourceBuffer->getBufferIdentifier().str());

    if (failed(visitor.walkAST()))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Failed to emit QUIR");
    // make sure to finish the in progress quir.circuit
    visitor.finishCircuit();
    if (mlir::failed(mlir::verify(newModule))) {
      newModule.dump();

      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Failed to verify generated QUIR");
    }
    qasm3ToMlirTiming.stop();
  }

  return llvm::Error::success();
} // parse
