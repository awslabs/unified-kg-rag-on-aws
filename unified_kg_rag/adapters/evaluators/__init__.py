# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluator base classes and the adapters wrapping scoring libraries.

Import each from its module (``base``, ``langchain_evaluator``,
``ragas_evaluator``): nothing is re-exported, so importing the base classes
does not load langchain's evaluators or ragas.
"""

import os

# ragas sends usage telemetry (an HTTP POST per evaluation) unless this is
# "true". Set before any module of this package can import ragas; a value the
# user set (RAGAS_DO_NOT_TRACK=false opts back in) is kept.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
