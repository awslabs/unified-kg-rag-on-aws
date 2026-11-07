# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compilers and static checkers of structured graph queries
(docs/design.md §6.6).

``gremlin`` compiles a validated plan to Gremlin bytecode (Neptune and
Gremlin Server), ``sqlite`` to parameterised SQL over the embedded graph,
and ``checks`` inspects the compiled form, independently of the
compilers, before anything is sent. The graph retrievers
(``adapters.retrievers``) run them on their own connections. Import the
modules themselves: this package imports nothing, so a module that needs
only one of them does not load the Gremlin client.
"""
