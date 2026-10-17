# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Content-guard adapters: the ``ContentGuardPort`` implementations.

- ``bedrock``: ``BedrockGuardrail``, which screens the user query and the
  final answer with the Amazon Bedrock ``ApplyGuardrail`` API
  (``aws.bedrock.guardrail``).

Nothing is re-exported here, so importing this package does not load boto3:
import from the submodules.
"""
