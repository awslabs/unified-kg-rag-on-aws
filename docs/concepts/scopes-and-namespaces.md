# Scopes and namespaces

> 🇰🇷 한국어판: [scopes-and-namespaces.ko.md](./scopes-and-namespaces.ko.md)

Two names decide where a run writes, what it may delete and what a query
reads: the **namespace** names a corpus's indexes, and the **source scope**
names the corpus source a run diffs within. Neither is an access control.

## Namespace

Every index and graph label of a corpus is named `<prefix>-<namespace>`
(plus `-<additional_suffix>` with `indexing.additional_suffix`). Ingestion
writes the namespace `processing.document_parsing.namespace` unless
`run-ingestion --namespace` names another; `run-rag`, `run-eval`,
`run-describe`, `run-visualization` and `run-mcp-server` read the same
namespace of their config unless `--namespace` names another, and in Python
`RAGInput.namespace` selects it (`None`: the chain config's). All default to
`default`. The console's Ingest page and its requests take a `namespace`
too. Use one namespace per tenant or corpus version (the same
`--namespace`, or the same config file, for ingestion and queries), with
the same `indexing.additional_suffix` on both sides; incremental indexing
and deletion are scoped to it ([Incremental indexing](../guides/incremental.md)).
Lowercase letters, digits, `-` and `_`.

The namespace had two older names, which still work as deprecated aliases
with a warning: the *index suffix* (`--suffix`, `RAGInput.suffix`,
`run-eval --arm ...,suffix=`) and the *index value*
(`processing.document_parsing.index_value`). Giving an old and the new
name two different values is refused.

Two index prefixes, or the two label prefixes, that could name one store
(equal, or one is the other followed by `-`, as `kg` and `kg-entities`)
fail the load.

With the doc-status registry, one run writes a namespace at a time (it
holds the namespace's write lease); different namespaces can run side by
side. The lease and the registry partition are keyed by the *index
namespace*: the namespace plus `indexing.additional_suffix`, plus a tag of
non-default `indexing.opensearch.*_index_prefix` /
`indexing.neptune.*_label_prefix` values, which the CLIs print as the
run's **Index Namespace** and log and error messages name
(`Index namespace '<namespace>' is being written by ...`). The lease,
takeover after a crash and what runs without the registry:
[Operator Runbook: Concurrent runs](../operations.md#concurrent-runs).

## Source scope

A run diffs within its **scope**: its namespace (`document_parsing.namespace`,
plus `indexing.additional_suffix`) together with its **source scope**, the
corpus source (`document_parsing.source_scope`, by default the resolved source directory;
the container entrypoint sets it to the S3 URI it syncs from; with and
without a trailing slash is one source scope, and a WARNING names the spelling
used). A document's
registry key is its scope and its path relative to the source directory, and a
run only diffs, and treats as deleted, the documents of its own scope. So
another tenant's run (another namespace) never deletes this tenant's
documents, a subfolder run as its own source directory never deletes the rest
of the corpus (its files register as separate documents: do not index the
same files from two roots into one namespace), and two sources indexed into one
namespace stay separate even for the same relative path. Document and text-unit
ids derive from the scope too, so set `source_scope` for ids that are
reproducible across paths and machines (the same corpus indexed from another
directory or host otherwise gets other ids).

The namespace is part of those ids as well, and a build step that orders by
id orders by it too: the summary tree's leaf walk and the initial centers of
its clustering (picked by an id hash), and the order of an entity's merged
description lines (the input of its description summary). So one corpus built into two namespaces,
or from two source scopes, is two builds rather than two copies: its tree
and some of its descriptions differ. Compare configurations on one
namespace (an A/A arm measures the noise there), not against a copy built
under another name.

Keeping a corpus at one source, moving it, and what a mass deletion asks:
[Incremental indexing: Deletion scope](../guides/incremental.md#deletion-scope).

## Not an access-control boundary

The namespace separates names, not access: whoever can query can name any
namespace. A multi-tenant service must derive it from the authenticated
caller on the server.

**Filters are not an access-control boundary** either: a store that does not declare
a key returns its content unfiltered, and an entity, relationship or
community report drawn from several documents passes when one of them
matches, so it can carry excluded documents' content. An item from several
documents also matches each key of a multi-key filter independently: the
seeds, relationships, claims and community reports carry each key's values of
all their documents, so `attr_category:x attr_grp:b` matches one whose
documents are `{category: x, grp: a}` and `{category: y, grp: b}`, although
no document is both. Separate tenants or users by
[namespace](#namespace), chosen server-side.

To isolate tenants in the infrastructure, give queries their own read-only
role, scope a per-tenant OpenSearch role to the tenant's indexes, and give a
tenant whose graph others must not read its own Neptune cluster:
[Operator Runbook: Query access and tenant isolation](../operations.md#query-access-and-tenant-isolation).
