#include "API/api.h"

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void addEnumValues(py::module &m) {
    py::enum_<qssc::ErrorCategory>(m, "ErrorCategory", py::arithmetic())
        .value("OpenQASM3ParseFailure",
                qssc::ErrorCategory::OpenQASM3ParseFailure)
        .value("QSSCompilerError", qssc::ErrorCategory::QSSCompilerError)
        .value("QSSCompilerNoInputError", qssc::ErrorCategory::QSSCompilerNoInputError)
        .value("QSSCompilerCommunicationFailure", qssc::ErrorCategory::QSSCompilerCommunicationFailure)
        .value("QSSCompilerEOFFailure", qssc::ErrorCategory::QSSCompilerEOFFailure)
        .value("QSSCompilerNonZeroStatus", qssc::ErrorCategory::QSSCompilerNonZeroStatus)
        .value("QSSCompilationFailure", qssc::ErrorCategory::QSSCompilationFailure)
        .value("QSSLinkerNotImplemented", qssc::ErrorCategory::QSSLinkerNotImplemented)
        .value("QSSLinkSignatureWarning", qssc::ErrorCategory::QSSLinkSignatureWarning)
        .value("QSSLinkSignatureError", qssc::ErrorCategory::QSSLinkSignatureError)
        .value("QSSLinkAddressError", qssc::ErrorCategory::QSSLinkAddressError)
        .value("QSSLinkSignatureNotFound", qssc::ErrorCategory::QSSLinkSignatureNotFound)
        .value("QSSLinkArgumentNotFoundWarning", qssc::ErrorCategory::QSSLinkArgumentNotFoundWarning)
        .value("QSSLinkInvalidPatchTypeError", qssc::ErrorCategory::QSSLinkInvalidPatchTypeError)
        .value("UncategorizedError", qssc::ErrorCategory::UncategorizedError)
        .export_values();
}
