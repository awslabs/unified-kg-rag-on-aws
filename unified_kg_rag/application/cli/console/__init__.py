# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The local web console of ``run-console`` (docs/design.md#101-the-console-run-console-opt-in).

An entry point like the CLIs: a FastAPI app (:mod:`.server`) over the
application layer (``GraphRAGChain``, ``get_config``), the CLIs as child
processes for the long jobs (:mod:`.jobs`) and the files those write. Needs
the ``console`` extra; nothing here loads on import, so ``run-console`` can
check the extra first.
"""
