# Scopes and namespaces

> 🇰🇷 한국어판: [scopes-and-namespaces.ko.md](./scopes-and-namespaces.ko.md)

Three names decide where a run writes, what it may delete and what a query
reads: the **index suffix** names a corpus's indexes, the **source scope**
names the corpus source a run diffs within, and the **namespace** is the set
of indexes one writing run holds at a time. None of them is an access
control.

## Index suffix

Every index and graph label of a corpus is named `<prefix>-<suffix>`
(plus `-<additional_suffix>` with `indexing.additional_suffix`); `<suffix>` is
the **index suffix**, and the CLIs print the result as the run's **Index
Namespace**. Ingestion takes it from `processing.document_parsing.index_value`
unless `run-ingestion --suffix` names another; `run-rag` and `run-eval`
query the same `index_value` of their config unless `--suffix` names another,
and in Python `RAGInput.suffix` selects it (the chain does not read
`index_value`). All default to `default`. Use one index suffix per
tenant or corpus version (the same `--suffix`, or the same config file, for
ingestion and queries), with the same `indexing.additional_suffix` on both
sides; incremental indexing and deletion are scoped to it
([Incremental indexing](../guides/incremental.md)).

## Source scope

A run diffs within its **scope**: its index suffix (`document_parsing.index_value`,
plus `indexing.additional_suffix`) together with its **source scope**, the
corpus source (`document_parsing.source_scope`, by default the resolved source directory;
the container entrypoint sets it to the S3 URI it syncs from; with and
without a trailing slash is one source scope, and a WARNING names the spelling
used). A document's
registry key is its scope and its path relative to the source directory, and a
run only diffs, and treats as deleted, the documents of its own scope. So
another tenant's run (another `index_value`) never deletes this tenant's
documents, a subfolder run as its own source directory never deletes the rest
of the corpus (its files register as separate documents: do not index the
same files from two roots into one suffix), and two sources indexed into one
suffix stay separate even for the same relative path. Document and text-unit
ids derive from the scope too, so set `source_scope` for ids that are
reproducible across paths and machines (the same corpus indexed from another
directory or host otherwise gets other ids).

Keeping a corpus at one source, moving it, and what a mass deletion asks:
[Incremental indexing: Deletion scope](../guides/incremental.md#deletion-scope).

## Namespace

The **namespace** is the index suffix (`index_value`, `default` when unset)
plus `indexing.additional_suffix`, plus a tag of non-default
`indexing.opensearch.*_index_prefix` / `indexing.neptune.*_label_prefix`
values; log and error messages call it the *index namespace*
(`Index namespace '<namespace>' is being written by ...`). With the
doc-status registry, one run writes a namespace at a time
(it holds the namespace's write lease); different namespaces can run side
by side. The lease, takeover after a crash and what runs without the
registry: [Operator Runbook: Concurrent runs](../operations.md#concurrent-runs).

## Not an access-control boundary

The suffix separates names, not access: whoever can query can name any
suffix. A multi-tenant service must derive it from the authenticated caller
on the server.

**Filters are not an access-control boundary** either: a store that does not declare
a key returns its content unfiltered, and an entity, relationship or
community report drawn from several documents passes when one of them
matches, so it can carry excluded documents' content. An item from several
documents also matches each key of a multi-key filter independently: the
seeds, relationships, claims and community reports carry each key's values of
all their documents, so `attr_category:x attr_grp:b` matches one whose
documents are `{category: x, grp: a}` and `{category: y, grp: b}`, although
no document is both. Separate tenants or users by
[index suffix](#index-suffix), chosen server-side.

To isolate tenants in the infrastructure, give queries their own read-only
role, scope a per-tenant OpenSearch role to the tenant's indexes, and give a
tenant whose graph others must not read its own Neptune cluster:
[Operator Runbook: Query access and tenant isolation](../operations.md#query-access-and-tenant-isolation).
