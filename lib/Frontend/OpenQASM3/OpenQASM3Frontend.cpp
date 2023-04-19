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

#include "Frontend/OpenQASM3/PrintQASM3Visitor.h"
#include "Frontend/OpenQASM3/QUIRGenQASM3Visitor.h"

#include "Dialect/QUIR/IR/QUIRDialect.h"

#include "qasm/Frontend/QasmDiagnosticEmitter.h"
#include "qasm/Frontend/QasmParser.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include <mutex>
#include <sstream>

static llvm::cl::OptionCategory openqasm3Cat(
    " OpenQASM 3 Frontend Options",
    "Options that control the OpenQASM 3 frontend of QSS Compiler");

static llvm::cl::opt<uint>
    numShots("num-shots",
             llvm::cl::desc("The number of shots to execute on the quantum "
                            "circuit, default is 1000"),
             llvm::cl::init(1000), llvm::cl::cat(openqasm3Cat));

static llvm::cl::opt<std::string> shotDelay(
    "shot-delay",
    llvm::cl::desc("Repetition delay between shots. Defaults to 1ms."),
    llvm::cl::init("1ms"), llvm::cl::cat(openqasm3Cat));

static llvm::cl::list<std::string>
    includeDirs("I", llvm::cl::desc("Add <dir> to the include path"),
                llvm::cl::value_desc("dir"), llvm::cl::cat(openqasm3Cat));

static qssc::DiagnosticCallback *diagnosticCallback_;
static llvm::SourceMgr *sourceMgr_;

static std::mutex qasmParserLock;

llvm::Error qssc::frontend::openqasm3::parse(
    std::string const &source, bool sourceIsFilename, bool emitRawAST,
    bool emitPrettyAST, bool emitMLIR, mlir::ModuleOp &newModule,
    llvm::Optional<qssc::DiagnosticCallback> diagnosticCallback) {

  // The QASM parser can only be called from a single thread.
  std::lock_guard<std::mutex> qasmParserLockGuard(qasmParserLock);

  for (const auto &dirStr : includeDirs)
    QASM::QasmPreprocessor::Instance().AddIncludePath(dirStr);

  QASM::ASTParser parser;
  QASM::ASTRoot *root = nullptr;
  llvm::SourceMgr sourceMgr;

  // Add a callback for diagnostics to the parser. Since the callback needs
  // access to diagnosticCallback to forward diagnostics, make it available in a
  // global variable.
  diagnosticCallback_ =
      diagnosticCallback.hasValue() ? diagnosticCallback.getPointer() : nullptr;
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
          qssc::Diagnostic diag{
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
    if (sourceIsFilename) {
      QASM::QasmPreprocessor::Instance().SetTranslationUnit(source);

      std::string errorMessage;
      auto file = mlir::openInputFile(source, &errorMessage);

      if (!file)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "Failed to open input file: " +
                                           errorMessage);

      sourceMgr.AddNewSourceBuffer(std::move(file), llvm::SMLoc());

      root = parser.ParseAST();

    } else {
      auto sourceBuffer = llvm::MemoryBuffer::getMemBuffer(source, "", false);

      sourceMgr.AddNewSourceBuffer(std::move(sourceBuffer), llvm::SMLoc());
      root = parser.ParseAST(source);
    }
  } catch (std::exception &e) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        llvm::Twine{"Exception while parsing OpenQASM 3 input: "} + e.what());
  }

  if (root == nullptr)
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
    auto *context = newModule.getContext();

    context->loadDialect<mlir::quir::QUIRDialect>();
    context->loadDialect<mlir::complex::ComplexDialect>();
    context->loadDialect<mlir::StandardOpsDialect>();

    mlir::OpBuilder builder(newModule.getBodyRegion());

    QASM::ASTStatementList *statementList =
        QASM::ASTStatementBuilder::Instance().List();

    qssc::frontend::openqasm3::QUIRGenQASM3Visitor visitor(builder, newModule,
                                                           /*filename=*/"");
    visitor.initialize(numShots, shotDelay);
    visitor.setStatementList(statementList);
    visitor.setInputFile(sourceIsFilename ? source : "-");

    if (failed(visitor.walkAST()))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Failed to emit QUIR");
    visitor.finalizeCircuit();
    if (mlir::failed(mlir::verify(newModule))) {
      newModule.dump();

      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Failed to verify generated QUIR");
    }
  }

  return llvm::Error::success();
}
