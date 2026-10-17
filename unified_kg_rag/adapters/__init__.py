# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapters layer: concrete implementations binding ports to technologies.

Two kinds of package. Vendor packages (``aws``, ``local``) hold technology
bindings: transport clients, embedded engines and stores, model factories
(LLM, embedding, rerank), token counters and caches. Role packages hold the
port implementations the application wires: ``storage`` (``GraphIndexer`` /
``VectorIndexer``), ``retrievers`` (graph and vector retrievers) and
``doc_status`` (``DocStatusPort``). A role package imports a vendor package,
never the other way round.
"""
