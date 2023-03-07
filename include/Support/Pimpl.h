//===- Pimpl.h - Pointer-to-implementation pointer wrapper ------*- C++ -*-===//
//
// (C) Copyright IBM 2021, 2023.
//
// This code is part of Qiskit.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
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
