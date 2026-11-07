# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public benchmarks prepared as a corpus plus a ``run-eval`` dataset.

The data is downloaded (or read from a local copy) at run time and never
committed to this repository.
"""

# The question classes of the assertions benchmark (``autoq``), here so the
# prepare-benchmark parser lists them without importing its generator.
QUESTION_CLASSES = (
    "data_local",
    "data_global",
    "data_linked",
    "activity_local",
    "activity_global",
)
