# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tracing adapters for the ``shared.observability`` port.

Nothing is imported here: ``otel`` needs the ``otel`` extra and is loaded only
when observability is enabled, and ``langchain_callback`` is imported by the
model factories that attach it.
"""
