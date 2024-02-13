#include "lib_enums.h"
#include "errors.h"

#include <pybind11/attr.h>
#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace py = pybind11;

void addErrorCategory(py::module &m) {
  py::enum_<qssc::ErrorCategory>(m, "ErrorCategory", py::arithmetic())
      .value("OpenQASM3ParseFailure",
             qssc::ErrorCategory::OpenQASM3ParseFailure)
      .value("QSSCompilerError", qssc::ErrorCategory::QSSCompilerError)
      .value("QSSCompilerNoInputError",
             qssc::ErrorCategory::QSSCompilerNoInputError)
      .value("QSSCompilerCommunicationFailure",
             qssc::ErrorCategory::QSSCompilerCommunicationFailure)
      .value("QSSCompilerEOFFailure",
             qssc::ErrorCategory::QSSCompilerEOFFailure)
      .value("QSSCompilerNonZeroStatus",
             qssc::ErrorCategory::QSSCompilerNonZeroStatus)
      .value("QSSCompilerSequenceTooLong",
             qssc::ErrorCategory::QSSCompilerSequenceTooLong)
      .value("QSSCompilationFailure",
             qssc::ErrorCategory::QSSCompilationFailure)
      .value("QSSLinkerNotImplemented",
             qssc::ErrorCategory::QSSLinkerNotImplemented)
      .value("QSSLinkSignatureWarning",
             qssc::ErrorCategory::QSSLinkSignatureWarning)
      .value("QSSLinkSignatureError",
             qssc::ErrorCategory::QSSLinkSignatureError)
      .value("QSSLinkAddressError", qssc::ErrorCategory::QSSLinkAddressError)
      .value("QSSLinkSignatureNotFound",
             qssc::ErrorCategory::QSSLinkSignatureNotFound)
      .value("QSSLinkArgumentNotFoundWarning",
             qssc::ErrorCategory::QSSLinkArgumentNotFoundWarning)
      .value("QSSLinkInvalidPatchTypeError",
             qssc::ErrorCategory::QSSLinkInvalidPatchTypeError)
      .value("QSSControlSystemResourcesExceeded",
             qssc::ErrorCategory::QSSControlSystemResourcesExceeded)
      .value("UncategorizedError", qssc::ErrorCategory::UncategorizedError)
      .export_values();
}

void addSeverity(py::module &m) {
  py::enum_<qssc::Severity>(m, "Severity")
      .value("Info", qssc::Severity::Info)
      .value("Warning", qssc::Severity::Warning)
      .value("Error", qssc::Severity::Error)
      .value("Fatal", qssc::Severity::Fatal)
      .export_values();
}

void addDiagnostic(py::module &m) {
  py::class_<qssc::Diagnostic>(m, "Diagnostic")
      .def_readonly("severity", &qssc::Diagnostic::severity)
      .def_readonly("category", &qssc::Diagnostic::category)
      .def_readonly("message", &qssc::Diagnostic::message)
      .def("__str__", &qssc::Diagnostic::toString)
      .def(py::pickle(
          [](const qssc::Diagnostic &d) {
            // __getstate__ serializes the C++ object into a tuple
            return py::make_tuple(d.severity, d.category, d.message);
          },
          [](py::tuple const &t) {
            // __setstate__ restores the C++ object from a tuple
            if (t.size() != 3)
              throw std::runtime_error("invalid state for unpickling");

            auto severity = t[0].cast<qssc::Severity>();
            auto category = t[1].cast<qssc::ErrorCategory>();
            auto message = t[2].cast<std::string>();

            return qssc::Diagnostic(severity, category, std::move(message));
          }));
}
