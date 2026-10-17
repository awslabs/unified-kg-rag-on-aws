# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Document-status registry adapters: the ``DocStatusPort`` implementations.

- ``dynamodb``: ``DynamoDBDocStatusStore``, the registry on a DynamoDB table
  (``providers.doc_status.kind: dynamodb``).
- ``sqlite``: ``SQLiteDocStatusStore``, the registry in one SQLite file
  (``providers.doc_status.kind: sqlite``).

Both conform structurally to ``unified_kg_rag.ports.DocStatusPort`` and pass
``tests/integration/test_doc_status_conformance.py``. The provider registry
(``adapters.providers``) builds them from ``config.providers.doc_status``.
Nothing is re-exported here, so importing this package loads neither boto3
nor SQLite: import from the submodules.
"""
