# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local (non-AWS) adapters for local mode (``backend: local``).

- ``models``: language-model factories for OpenAI-compatible servers and Ollama.
- ``embeddings``: embedding factories for the same two servers.
- ``rerank``: a sentence-transformers cross-encoder reranker and the no-op one.
- ``token_counter``: a token counter that needs no model API.
- ``sqlite_doc_status``: the document-status registry on SQLite.
- ``search_engine``: an embedded vector/lexical store on SQLite that answers
  the OpenSearch queries of ``OpenSearchIndexer`` and ``OpenSearchRetriever``.

Each conforms structurally to its port (``ModelFactoryPort``,
``TokenCounterPort``, ``DocStatusPort``). The provider registry
(``adapters.provider_registry``) builds them from ``config.providers``; the
third-party clients of the ``[local]`` extra are imported only when a model is
built, so importing this package needs none of them. Nothing is re-exported
here, for the same reason: import from the submodules.
"""
