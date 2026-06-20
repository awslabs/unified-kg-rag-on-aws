# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Hexagonal (ports & adapters) interfaces owned by the domain core.

A *port* is an abstract interface the domain depends on; *adapters* are the
concrete implementations binding it to a technology. Ports live here so domain
code never imports a concrete backend.

``DocStatusPort`` is the document-status registry boundary used by incremental
indexing (adapter: ``aws_graphrag.adapters.aws.dynamodb``; in-memory fake in tests).

The write-side store contracts (full + delta) live as ABCs in
``aws_graphrag.storage.base`` (``GraphIndexer`` / ``VectorIndexer``) and the
read-side retrieval contract is ``aws_graphrag.retrieval.base.BaseGraphRAGRetriever`` —
both are real, implemented ports; they are kept next to their adapters rather
than duplicated here.
"""

from .doc_status import DocStatusPort

__all__ = ["DocStatusPort"]
