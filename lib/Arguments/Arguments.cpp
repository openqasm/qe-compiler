//===- Arguments.cpp -------------------------------------------*- C++ -*-===//
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
///  This file implements support for updating argument values after compilation
///
//===----------------------------------------------------------------------===//

#include "Arguments/Arguments.h"
#include "API/errors.h"
#include "Arguments/Signature.h"
#include "Payload/Payload.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace qssc::arguments {

using namespace payload;

llvm::Error updateParameters(qssc::payload::PatchablePayload *payload,
                             Signature &sig, ArgumentSource const &arguments,
                             bool treatWarningsAsErrors,
                             BindArgumentsImplementationFactory &factory,
                             const OptDiagnosticCallback &onDiagnostic,
                             int numberOfThreads) {

  std::deque<std::thread> threads;
  std::vector<std::shared_ptr<BindArgumentsImplementation>> binaries;

  bool const enableThreads = (numberOfThreads != 0);
  uint MAX_NUM_THREADS = (numberOfThreads > 0)
                             ? numberOfThreads
                             : std::thread::hardware_concurrency();

  // if failed to detect number of CPUs default to 10
  if (MAX_NUM_THREADS == 0)
    MAX_NUM_THREADS = 10;

  std::mutex errorMutex;
  bool errorSet = false;
  llvm::Error firstError = llvm::Error::success();

  // the onDiagnastic method used to emit diagnostics to python
  // is not thread safe
  // setup of local callback to capture the highest level diagnostic
  // and re-emit from the main thread if threading is being used
  std::optional<qssc::Diagnostic> localDiagValue = std::nullopt;
  std::optional<DiagnosticCallback> const localCallback =
      std::optional(std::function([&](const Diagnostic &diag) {
        if (!localDiagValue.has_value() ||
            localDiagValue.value().severity < diag.severity) {
          localDiagValue = diag;
        }
      }));

  uint numThreads = 0;
  for (const auto &[binaryName, patchPoints] : sig.patchPointsByBinary) {

    if (patchPoints.size() == 0) // no patch points
      continue;

    auto binaryDataOrErr = payload->readMember(binaryName);

    if (!binaryDataOrErr) {
      auto error = binaryDataOrErr.takeError();
      return emitDiagnostic(onDiagnostic, qssc::Severity::Error,
                            qssc::ErrorCategory::QSSLinkSignatureError,
                            "Error reading " + binaryName + " " +
                                toString(std::move(error)));
    }

    auto &binaryData = binaryDataOrErr.get();

    // onDiagnostic callback is not thread safe
    auto localDiagnostic = (enableThreads) ? localCallback : onDiagnostic;

    auto binary = std::shared_ptr<BindArgumentsImplementation>(
        factory.create(binaryData, localDiagnostic));
    binary->setTreatWarningsAsErrors(treatWarningsAsErrors);

    if (enableThreads) {
      // save shared point in vector to ensure lifetime exceeds the thread
      binaries.emplace_back(binary);

      numThreads++;
      if (numThreads > MAX_NUM_THREADS) {
        // wait for a thread to finish before starting another
        auto &t = threads[0];
        t.join();
        threads.pop_front();
      }
      threads.emplace_back([&, binary] {
        if (errorSet)
          return;

        for (auto const &patchPoint : patchPoints) {
          auto err = binary->patch(patchPoint, arguments);
          if (err && !errorSet) {
            const std::lock_guard<std::mutex> lock(errorMutex);
            firstError = std::move(err);
            errorSet = true;
            return;
          }
        }
      });
    } else {
      // processing patch points on main thread
      for (auto const &patchPoint : patchPoints)
        if (auto err = binary->patch(patchPoint, arguments))
          return err;
    }
  }

  if (enableThreads) {
    for (auto &t : threads)
      t.join();

    binaries.clear();

    if (errorSet || localDiagValue.has_value()) {
      // emit error or warning via onDiagnostic if
      // one was set
      auto *diagnosticCallback =
          onDiagnostic.has_value() ? &onDiagnostic.value() : nullptr;
      if (diagnosticCallback && localDiagValue.has_value())
        (*diagnosticCallback)(localDiagValue.value());
      // possibly return the error
      auto minLevel =
          (treatWarningsAsErrors) ? Severity::Info : Severity::Warning;
      if (localDiagValue.has_value() &&
          (localDiagValue.value().severity > minLevel)) {
        return firstError;
      }
    }
  }

  return llvm::Error::success();
}

llvm::Error
bindArguments(llvm::StringRef moduleInput, llvm::StringRef payloadOutputPath,
              ArgumentSource const &arguments, bool treatWarningsAsErrors,
              bool enableInMemoryInput, std::string *inMemoryOutput,
              BindArgumentsImplementationFactory &factory,
              const OptDiagnosticCallback &onDiagnostic, int numberOfThreads) {

  bool const enableInMemoryOutput = payloadOutputPath == "";

  // placeholder string for data on disk if required
  std::string inputFromDisk;

  if (!enableInMemoryInput) {
    // compile payload on disk
    // copy to link payload if not returning in memory
    // load from disk into string if returning in memory
    if (!enableInMemoryOutput) {
      std::error_code const copyError =
          llvm::sys::fs::copy_file(moduleInput, payloadOutputPath);
      if (copyError)
        return llvm::make_error<llvm::StringError>(
            "Failed to copy circuit module to payload", copyError);
    } else {
      // read from disk to process in memory
      std::ostringstream buf;
      std::ifstream const input(moduleInput.str().c_str());
      buf << input.rdbuf();
      inputFromDisk = buf.str();
      moduleInput = inputFromDisk;
      enableInMemoryInput = true;
    }
  }

  if (!enableInMemoryOutput && enableInMemoryInput) {
    // if payload in memory but returning on disk
    // copy to disk and process from there
    std::ofstream payload;
    payload.open(payloadOutputPath.str(), std::ios::binary);
    payload.write(moduleInput.str().c_str(), moduleInput.str().length());
    payload.close();
    enableInMemoryInput = false;
  }

  llvm::StringRef const payloadData =
      (enableInMemoryInput) ? moduleInput : payloadOutputPath;

  auto binary = std::unique_ptr<BindArgumentsImplementation>(
      factory.create(onDiagnostic));
  binary->setTreatWarningsAsErrors(treatWarningsAsErrors);

  auto payload = std::unique_ptr<PatchablePayload>(
      binary->getPayload(payloadData, enableInMemoryInput));

  auto sigOrError = binary->parseSignature(payload.get());
  if (auto err = sigOrError.takeError())
    return err;

  if (auto err = updateParameters(payload.get(), sigOrError.get(), arguments,
                                  treatWarningsAsErrors, factory, onDiagnostic,
                                  numberOfThreads))
    return err;

  // setup linked payload I/O
  // if enableInMemoryOutput is true:
  //    write to string
  // if enableInMemoryInput is true:
  //    payload is not on disk yet, do not assume payload->writeBack()
  //    will write the full payload to disk so: write to string,
  //    dump string to disk and clear string
  // if enableInMemoryInput is false:
  //    payload was on disk originally use writeBack
  if (auto err = payload->writeBack())
    return err;
  if (enableInMemoryOutput || enableInMemoryInput) {
    if (auto err = payload->writeString(inMemoryOutput))
      return err;
    if (!enableInMemoryOutput) {
      auto pathStr = payloadOutputPath.str();
      std::ofstream out(pathStr);
      out << inMemoryOutput;
      out.close();
      // clear output string
      *inMemoryOutput = "";
    }
  }

  return llvm::Error::success();
}

} // namespace qssc::arguments
