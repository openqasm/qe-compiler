//===- errors.cpp  - Error reporting API ------------------------*- C++ -*-===//
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
///  This file implements the API for error reporting.
///
//===----------------------------------------------------------------------===//

#include "API/errors.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>

namespace {

std::string_view getErrorCategoryAsString(qssc::ErrorCategory category) {
  using namespace qssc;
  switch (category) {
  case ErrorCategory::OpenQASM3ParseFailure:
    return "OpenQASM 3 parse error";

  case ErrorCategory::OpenQASM3UnsupportedInput:
    return "OpenQASM 3 semantics are unsupported";

  case ErrorCategory::QSSCompilerError:
    return "Unknown compiler error";

  case ErrorCategory::QSSCompilerNoInputError:
    return "Error when no input file or string is provided";

  case ErrorCategory::QSSCompilerCommunicationFailure:
    return "Error on compilation communication failure";

  case ErrorCategory::QSSCompilerEOFFailure:
    return "EOF Error";

  case ErrorCategory::QSSCompilerNonZeroStatus:
    return "Errored because non-zero status is returned";

  case ErrorCategory::QSSCompilerSequenceTooLong:
    return "Input sequence is too long";

  case ErrorCategory::QSSCompilationFailure:
    return "Failure during compilation";

  case ErrorCategory::QSSLinkerNotImplemented:
    return "BindArguments not implemented for target";

  case ErrorCategory::QSSLinkSignatureWarning:
    return "Signature file format is invalid but may be processed";

  case ErrorCategory::QSSLinkSignatureError:
    return "Signature file format is invalid";

  case ErrorCategory::QSSLinkAddressError:
    return "Signature address is invalid";

  case ErrorCategory::QSSLinkSignatureNotFound:
    return "Signature file not found";

  case ErrorCategory::QSSLinkArgumentNotFoundWarning:
    return "Parameter in signature not found in arguments";

  case ErrorCategory::QSSLinkInvalidPatchTypeError:
    return "Invalid patch point type";

  case ErrorCategory::QSSControlSystemResourcesExceeded:
    return "Control system resources exceeded";

  case ErrorCategory::QSSTargetUnsupportedOperation:
    return "An unsupported operation on a target was used";

  case ErrorCategory::UncategorizedError:
    return "Compilation failure";
  }

  llvm_unreachable("unhandled category");
}

llvm::StringRef getSeverityAsString(qssc::Severity sev) {
  switch (sev) {
  case qssc::Severity::Info:
    return "Info";
  case qssc::Severity::Warning:
    return "Warning";
  case qssc::Severity::Error:
    return "Error";
  case qssc::Severity::Fatal:
    return "Fatal";
  }

  llvm_unreachable("unhandled severity");
}

} // anonymous namespace

namespace qssc {

std::string Diagnostic::toString() const {
  std::string str;
  llvm::raw_string_ostream ostream(str);

  ostream << getSeverityAsString(severity) << ": "
          << getErrorCategoryAsString(category) << "\n";
  ostream << message;

  return str;
}

llvm::Error emitDiagnostic(const OptDiagnosticCallback &onDiagnostic,
                           const Diagnostic &diag, std::error_code ec) {
  auto *diagnosticCallback =
      onDiagnostic.has_value() ? &onDiagnostic.value() : nullptr;
  if (diagnosticCallback)
    (*diagnosticCallback)(diag);
  return llvm::createStringError(ec, diag.toString());
}

llvm::Error emitDiagnostic(const OptDiagnosticCallback &onDiagnostic,
                           Severity severity, ErrorCategory category,
                           std::string message, std::error_code ec) {
  qssc::Diagnostic const diag{severity, category, std::move(message)};
  return emitDiagnostic(onDiagnostic, diag, ec);
}

namespace {
std::string ErrorCategoryAttrName = "QSSCErrorCategory";

std::optional<ErrorCategory> getQSSCDiagnosticCategory(mlir::DiagnosticArgument &arg) {
  if (arg.getKind() ==
      mlir::DiagnosticArgument::DiagnosticArgumentKind::Attribute) {
    if (auto dictAttr =
            arg.getAsAttribute().dyn_cast_or_null<mlir::DictionaryAttr>()) {
      if (auto namedAttr = dictAttr.getNamed(ErrorCategoryAttrName)) {
        if (auto catInt = namedAttr.value()
                              .getValue()
                              .dyn_cast<mlir::IntegerAttr>()) {
          auto cat = static_cast<ErrorCategory>((uint32_t)catInt.getInt());
          return cat;
        }

        llvm_unreachable("Invalid attribute type for error category. Must "
                          "be an integer.");
      }
    }
  }
  return std::nullopt;
}

// We decode the QSSC ErrorCategory as an attribute argument that a dictionary
// attribute containing a named attribute that is an integer encoding the error
// category enum value.
std::optional<ErrorCategory> lookupErrorCategory(mlir::Diagnostic &diagnostic) {
  for (auto &note : diagnostic.getNotes()) {
    for (auto &arg : note.getArguments()) {
      if (auto cat = getQSSCDiagnosticCategory(arg))
        return cat.value();
    }
  }
  // Not found
  return std::nullopt;
}

// Recursively populate diagnostic (including notes except those that are for the
// QSSC error category)
void populateDiagnosticIgnoringQSSC(mlir::Diagnostic &from, mlir::Diagnostic &to) {
    for (auto &arg: from.getArguments()) {
      // Do not copy diagnostic category
      if (!getQSSCDiagnosticCategory(arg).has_value())
        to.append(arg);
    }
    // Recurse on notes
    for (auto &fromNote: from.getNotes()) {
      // First past through args and ensure there are non-qssc notes
      // as we do not want to add an empty note.
      bool containsNonQSSCNote = false;
      for (auto &arg: fromNote.getArguments()) {
        // Do not copy diagnostic category
        if (!getQSSCDiagnosticCategory(arg).has_value())
          containsNonQSSCNote = true;
      }
      if (containsNonQSSCNote) {
        auto &toNote = to.attachNote(fromNote.getLocation());
        populateDiagnosticIgnoringQSSC(fromNote, toNote);
      }
    }
}

} // anonymous namespace

// We encode the QSSC ErrorCategory as an attribute argument that a dictionary
// attribute containing a named attribute that is an integer encoding the error
// category enum value.
void encodeQSSCError(mlir::MLIRContext *context,
                     mlir::InFlightDiagnostic &diagnostic,
                     ErrorCategory category) {
  return encodeQSSCError(context, diagnostic.getUnderlyingDiagnostic(),
                         category);
}

void encodeQSSCError(mlir::MLIRContext *context, mlir::Diagnostic *diagnostic,
                     ErrorCategory category) {
  auto builder = mlir::OpBuilder(context);
  mlir::StringAttr key = builder.getStringAttr(ErrorCategoryAttrName);
  mlir::IntegerAttr value =
      builder.getI32IntegerAttr(static_cast<int32_t>(category));
  auto attr = builder.getDictionaryAttr(builder.getNamedAttr(key, value));
  diagnostic->attachNote().append(attr);
}

std::optional<Diagnostic> decodeQSSCDiagnostic(mlir::Diagnostic &diagnostic) {
  // map diagnostic severity to qssc severity
  auto severity = diagnostic.getSeverity();
  qssc::Severity qsscSeverity = qssc::Severity::Error;
  switch (severity) {
  case mlir::DiagnosticSeverity::Error:
    qsscSeverity = qssc::Severity::Error;
    break;
  case mlir::DiagnosticSeverity::Warning:
    qsscSeverity = qssc::Severity::Warning;
    break;
  case mlir::DiagnosticSeverity::Note:
  case mlir::DiagnosticSeverity::Remark:
    qsscSeverity = qssc::Severity::Info;
  }

  auto errorCategory = lookupErrorCategory(diagnostic);
  if (errorCategory.has_value())
    return Diagnostic(qsscSeverity, errorCategory.value(), diagnostic.str());

  // Default error category for unspecified MLIR error diagnostics of Error
  // severity.
  if (qsscSeverity == qssc::Severity::Error)
    return Diagnostic(qsscSeverity, ErrorCategory::QSSCompilationFailure,
                      diagnostic.str());

  return std::nullopt;
}

mlir::InFlightDiagnostic emitError(mlir::Operation *op, ErrorCategory category,
                                   const llvm::Twine &message) {
  auto diagnostic = op->emitError(message);
  encodeQSSCError(op->getContext(), diagnostic, category);
  return diagnostic;
}

mlir::InFlightDiagnostic emitOpError(mlir::Operation *op,
                                     ErrorCategory category,
                                     const llvm::Twine &message) {
  auto diagnostic = op->emitOpError(message);
  encodeQSSCError(op->getContext(), diagnostic, category);
  return diagnostic;
}

mlir::InFlightDiagnostic emitRemark(mlir::Operation *op, ErrorCategory category,
                                    const llvm::Twine &message) {
  auto diagnostic = op->emitRemark(message);
  encodeQSSCError(op->getContext(), diagnostic, category);
  return diagnostic;
}

mlir::InFlightDiagnostic emitWarning(mlir::Operation *op,
                                     ErrorCategory category,
                                     const llvm::Twine &message) {
  auto diagnostic = op->emitWarning(message);
  encodeQSSCError(op->getContext(), diagnostic, category);
  return diagnostic;
}

QSSCMLIRDiagnosticHandler::QSSCMLIRDiagnosticHandler(
    llvm::SourceMgr &mgr, mlir::MLIRContext *ctx,
    const OptDiagnosticCallback &diagnosticCb)
    : mlir::SourceMgrDiagnosticHandler(mgr, ctx), diagnosticCb(diagnosticCb) {

      // Replace the source manager handler set through inheritance
      // with our own implementation. This will eventually call the
      // source manager handler.
      setHandler([&](mlir::Diagnostic &diag) { this->emitDiagnostic(diag); });
    }

void QSSCMLIRDiagnosticHandler::emitDiagnostic(mlir::Diagnostic &diagnostic) {
  // emit diagnostic cast to void to discard result as it is not needed here
  if (auto decoded = qssc::decodeQSSCDiagnostic(diagnostic))
    (void)qssc::emitDiagnostic(diagnosticCb, decoded.value());

  auto filteredDiagnostic = filterQSSCDiagnostic(diagnostic);
  mlir::SourceMgrDiagnosticHandler::emitDiagnostic(filteredDiagnostic);
}

mlir::Diagnostic QSSCMLIRDiagnosticHandler::filterQSSCDiagnostic(mlir::Diagnostic &diagnostic) {
  auto filteredDiagnostic = mlir::Diagnostic(diagnostic.getLocation(), diagnostic.getSeverity());

  populateDiagnosticIgnoringQSSC(diagnostic, filteredDiagnostic);

  return filteredDiagnostic;
}

} // namespace qssc
