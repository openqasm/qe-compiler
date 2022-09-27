# ===- lit.local.cfg.py --------------------------------------*- Python -*-===//
#
# (C) Copyright IBM 2022.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
#
# ===----------------------------------------------------------------------===//

config.substitutions.append(("%TEST_CFG", lit_config.params["TEST_CFG"]))
