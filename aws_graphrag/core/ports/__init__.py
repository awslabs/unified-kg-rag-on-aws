# Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties.
"""Hexagonal (ports & adapters) interfaces for aws-graphrag.

A *port* is an abstract interface owned by the domain core. *Adapters* are the
concrete implementations that bind a port to an external technology — Neptune,
OpenSearch, DynamoDB, S3, Bedrock. Domain and algorithm code depends only on
ports, never on a specific adapter, so backends can be swapped or faked (see
``tests/fixtures/fakes``) without touching business logic.

These ports are introduced incrementally (strangler pattern). ``DocStatusPort``
is exercised today via the in-memory fake; ``GraphStorePort`` and
``VectorStorePort`` declare the forward surface (incl. the upsert/delete methods
M2/M3 will add) that existing indexers will be migrated behind — they are not
yet wired into the concrete classes. They are defined as ``typing.Protocol`` so
classes conform structurally without inheriting, avoiding a base-class swap.
"""
from .doc_status import DocStatusPort
from .graph_store import GraphStorePort
from .vector_store import VectorStorePort

__all__ = [
    "DocStatusPort",
    "GraphStorePort",
    "VectorStorePort",
]
