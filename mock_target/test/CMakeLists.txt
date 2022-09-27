# (C) Copyright IBM 2021.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

qssc_add_lit_test_suite(check-mock
        "Mock Static"
        "${CMAKE_CURRENT_SOURCE_DIR}"
        "${LIT_TEST_EXTRA_ARGS} -v -DTEST_CFG=${CMAKE_CURRENT_SOURCE_DIR}/test.cfg"
)
