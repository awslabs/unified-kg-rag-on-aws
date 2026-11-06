# RAPTOR fidelity fixtures

`upstream.json` holds the answers of the original RAPTOR code
(github.com/parthsarthi03/raptor, MIT, commit `7da1d48`) on synthetic
inputs, for `tests/unit/test_raptor_fidelity.py`. Only the cases' parameters
and the original's answers are stored: the points come from
`tests/fixtures/raptor_fidelity.py` (plain Python, SHA-256 draws), on both
sides.

They were recorded in the original's own environment (numpy, scikit-learn,
scipy, tenacity; never a dependency of this package), with its model
clients stubbed, UMAP (unseeded in the original) replaced by an identity
truncation so both sides cluster the same points, and tiktoken by a word
count so both count the same tokens:

- `clustering`: `cluster_utils.GMM_cluster` (BIC over 1..49, soft labels at
  0.1) on 40 sets of 5-dimensional blobs of 50-90 points a blob;
- `two_step`: `cluster_utils.perform_clustering` (global, then local) on 10
  sets of 10-dimensional blobs of 100-130 points a blob, with the global
  count;
- `trees`: `ClusterTreeBuilder.construct_tree` on 12 leaf sets, with every
  clustering call it made (node indices and labels) and the children of
  each node it built, layer by layer;
- `selection`: `TreeRetriever.retrieve_information_collapse_tree` (top_k
  10, 3,500 tokens) on 40 node sets.

`test_the_recorded_answers_are_the_originals` checks them against a
checkout (`RAPTOR_UPSTREAM=<checkout> uv run --with scikit-learn pytest
tests/unit/test_raptor_fidelity.py`); it is skipped otherwise.
