//===- errors.h - Error Reporting API ---------------------------*- C++ -*-===//
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
///  This file declares the API for reporting errors.
///
//===----------------------------------------------------------------------===//

#ifndef QSS_COMPILER_ERROR_H
#define QSS_COMPILER_ERROR_H

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"

#include <functional>
#include <list>
#include <optional>
#include <string>

namespace qssc {

enum class ErrorCategory {
  OpenQASM3ParseFailure,
  QSSCompilerError,
  QSSCompilerNoInputError,
  QSSCompilerCommunicationFailure,
  QSSCompilerEOFFailure,
  QSSCompilerNonZeroStatus,
  QSSCompilerSequenceTooLong,
  QSSCompilationFailure,
  QSSLinkerNotImplemented,
  QSSLinkSignatureWarning,
  QSSLinkSignatureError,
  QSSLinkAddressError,
  QSSLinkSignatureNotFound,
  QSSLinkArgumentNotFoundWarning,
  QSSLinkInvalidPatchTypeError,
  QSSControlSystemResourcesExceeded,
  OpenQASM3UnsupportedInput,
  UncategorizedError,
};

enum class Severity {
  Info,
  Warning,
  Error,
  Fatal,
};

struct Diagnostic {
public:
  Severity severity;
  ErrorCategory category;
  std::string message; /// a detailed and actionable error message

  Diagnostic(Severity severity_, ErrorCategory category_, std::string message_)
      : severity(severity_), category(category_), message(std::move(message_)) {
  }

  std::string toString() const;
};

using DiagList = std::list<Diagnostic>;
using DiagRefList = std::list<std::reference_wrapper<const Diagnostic>>;
using DiagnosticCallback = std::function<void(const Diagnostic &)>;
using OptDiagnosticCallback = std::optional<DiagnosticCallback>;

llvm::Error emitDiagnostic(const OptDiagnosticCallback &onDiagnostic,
                           const Diagnostic &diag,
                           std::error_code ec = llvm::inconvertibleErrorCode());
llvm::Error emitDiagnostic(const OptDiagnosticCallback &onDiagnostic,
                           Severity severity, ErrorCategory category,
                           std::string message,
                           std::error_code ec = llvm::inconvertibleErrorCode());

/// Encode QSSC diagnostic information within the notes of an
/// MLIR diagnostic. This enables the usage of MLIR's diagnostic
/// mechanisms to return QSSC diagnostics. This information
/// will be extracted by the QSSC runtime API and handled like other
/// emitted diagnostics.
/// @param context The active context
/// @param diagnostic Inflight diagnostic to encode into
/// @param category The QSSC error category to encode. This will be encoded as a
/// diagnostic note.
void encodeQSSCError(mlir::MLIRContext *context,
                     mlir::InFlightDiagnostic &diagnostic,
                     ErrorCategory category);

/// Encode QSSC diagnostic information within the notes of an
/// MLIR diagnostic. This enables the usage of MLIR's diagnostic
/// mechanisms to return QSSC diagnostics. This information
/// will be extracted by the QSSC runtime API and handled like other
/// emitted diagnostics.
/// @param context The active context
/// @param diagnostic Diagnostic to encode into
/// @param category The QSSC error category to encode. This will be encoded as a
/// diagnostic note.
void encodeQSSCError(mlir::MLIRContext *context, mlir::Diagnostic *diagnostic,
                     ErrorCategory category);

/// Decode the MLIR diagnostic into a QSSC Diagnostic (if necessary). If the
/// diagnostic has a QSSC diagnostic encoded through encodeQSSCError the emitted
/// diagnostic will contain this information. If std::nullopt is returned no
/// QSSC diagnostic should be generated.
std::optional<Diagnostic> decodeQSSCDiagnostic(mlir::Diagnostic &diagnostic);

/// Emit a QSSC encoded MLIR error on an operation. Reporting up to any
/// diagnostic handlers that may be listening.
mlir::InFlightDiagnostic emitError(mlir::Operation *op, ErrorCategory category,
                                   const llvm::Twine &message = {});

/// Emit a QSSC encoded MLIR operation error on an operation. Reporting up to
/// any diagnostic handlers that may be listening.
mlir::InFlightDiagnostic emitOpError(mlir::Operation *op,
                                     ErrorCategory category,
                                     const llvm::Twine &message = {});

/// Emit a QSSC encoded MLIR remark on an operation. Reporting up to any
/// diagnostic handlers that may be listening.
mlir::InFlightDiagnostic emitRemark(mlir::Operation *op, ErrorCategory category,
                                    const llvm::Twine &message = {});

/// Emit a QSSC encoded MLIR warning on an operation. Reporting up to any
/// diagnostic handlers that may be listening.
mlir::InFlightDiagnostic emitWarning(mlir::Operation *op,
                                     ErrorCategory category,
                                     const llvm::Twine &message = {});

/// Diagnostic handler for the QSSC compiler which will emit MLIR diagnostics
/// through the compiler's diagnostic interface as well as through MLIR's
/// source manager handler.
class QSSCMLIRDiagnosticHandler : public mlir::ScopedDiagnosticHandler {
public:
  QSSCMLIRDiagnosticHandler(llvm::SourceMgr &mgr, mlir::MLIRContext *ctx,
                            const OptDiagnosticCallback &diagnosticCb);

private:
  const OptDiagnosticCallback &diagnosticCb;
  // Must be pointer as we need to initialize after this class to avoid
  // registering handle with MLIR context before this class.
  std::unique_ptr<mlir::SourceMgrDiagnosticHandler> sourceMgrDiagnosticHandler;

  void emitDiagnostic(mlir::Diagnostic &diagnostic);
};

} // namespace qssc

#endif // QSS_COMPILER_ERROR_H
