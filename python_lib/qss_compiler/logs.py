# (C) Copyright IBM 2023.
#
# This code is part of Qiskit.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logging setup and utilties for the qss-compiler."""

import io
import logging


class StreamLogger(io.StringIO):
    """Implement StringIO and writes to a provided logger."""

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self.buffer_ = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.flush()

    def write(self, msg: str):
        if msg.endswith("\n"):
            self.buffer_.append(msg.rstrip("\n"))
            self.flush()
        else:
            self.buffer_.append(msg)

    def flush(self):
        msg = "".join(self.buffer_)
        if msg:
            self.logger.log(self.level, msg)
        self.buffer_ = []
