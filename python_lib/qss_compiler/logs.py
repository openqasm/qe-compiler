# (C) Copyright IBM 2023.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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
