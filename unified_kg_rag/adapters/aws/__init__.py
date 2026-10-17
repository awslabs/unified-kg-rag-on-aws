# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""AWS technology bindings: the Bedrock model factories, the Neptune and
OpenSearch clients, and the S3 transfer and caches.

The port implementations on these services live in the role packages
(``adapters.storage``, ``adapters.retrievers``, ``adapters.doc_status``).

Import each from its module: nothing is re-exported, so importing one adapter
does not load the SDKs of the others.
"""
