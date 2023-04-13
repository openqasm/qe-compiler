# (C) Copyright IBM 2023.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Add QSSConfig lib
qssc_add_library(QSSConfig
    QSSConfig.cpp
    CLIConfig.cpp
    EnvVarConfig.cpp

    ADDITIONAL_HEADER_DIRS
    ${QSSC_INCLUDE_DIR}/Config
)
