//===- Pimpl.h - Pointer-to-implementation pointer wrapper ------*- C++ -*-===//
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
//
// Wraps a unique_ptr<T>, propagating const-ness.
// For use in the Pimpl idiom.
//
// https://en.cppreference.com/w/cpp/language/pimpl
//
//===----------------------------------------------------------------------===//

#ifndef QSS_OPT_PIMPL_H
#define QSS_OPT_PIMPL_H

#include <memory>

namespace qssc::support {

template <typename T>
class Pimpl {
public:
  explicit Pimpl(std::unique_ptr<T> impl) : impl(std::move(impl)) {}
  inline const T *operator->() const { return impl.get(); }
  inline T *operator->() { return impl.get(); }

private:
  std::unique_ptr<T> impl;
};

} // namespace qssc::support

#endif // QSS_OPT_PIMPL_H
