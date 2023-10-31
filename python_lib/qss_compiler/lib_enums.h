#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void addErrorCategory(py::module &m);

void addSeverity(py::module &m);

void addDiagnostic(py::module &m);
