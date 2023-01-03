//===- OpenQASM3Frontend.cpp ------------------------------------*- C++ -*-===//
//
// (C) Copyright IBM 2023.
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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

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

llvm::Error qssc::frontend::openqasm3::parse(std::string const &source,
                                             bool sourceIsFilename,
                                             bool emitRawAST,
                                             bool emitPrettyAST, bool emitMLIR,
                                             mlir::ModuleOp &newModule) {
  for (const auto &dirStr : includeDirs)
    QASM::QasmPreprocessor::Instance().AddIncludePath(dirStr);

  QASM::ASTParser parser;
  QASM::ASTRoot *root = nullptr;

  QASM::QasmDiagnosticEmitter::SetHandler(
      [](const std::string &Exp, const std::string &Msg,
         QASM::QasmDiagnosticEmitter::DiagLevel DL) {
        std::string level = "unknown";

        switch (DL) {
        case QASM::QasmDiagnosticEmitter::DiagLevel::Error:
          level = "Error";
          break;

        case QASM::QasmDiagnosticEmitter::DiagLevel::ICE:
          level = "ICE";
          break;

        case QASM::QasmDiagnosticEmitter::DiagLevel::Warning:
          level = "Warning";
          break;

        case QASM::QasmDiagnosticEmitter::DiagLevel::Info:
          level = "Info";
          break;

        case QASM::QasmDiagnosticEmitter::DiagLevel::Status:
          level = "Status";
          break;
        }

        llvm::errs() << level << " while parsing OpenQASM 3 input\n"
                     << Exp << " " << Msg << "\n";

        if (DL == QASM::QasmDiagnosticEmitter::DiagLevel::Error ||
            DL == QASM::QasmDiagnosticEmitter::DiagLevel::ICE) {
          // give up parsing after errors right away (TODO update to recent
          // qss-qasm to support continuing)
          throw std::runtime_error("Failure parsing");
        }
      });

  try {
    if (sourceIsFilename) {
      QASM::QasmPreprocessor::Instance().SetTranslationUnit(source);
      root = parser.ParseAST();

    } else {

      root = parser.ParseAST(source);
    }
  } catch (std::exception &e) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        llvm::Twine{"Exception while parsing OpenQASM 3 input: "} + e.what());
  }

  assert(root != nullptr);

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
    if (mlir::failed(mlir::verify(newModule))) {
      newModule.dump();

      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "Failed to verify generated QUIR");
    }
  }

  return llvm::Error::success();
}
