# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-strategy search settings (global, local, DRIFT, LightRAG, PPR, tree, ...)."""

from typing import Annotated, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
)

from ..retrieval import (
    SearchStrategy,
    SearchType,
)


class GlobalSearchConfig(BaseModel):
    """MS GraphRAG global search: map over a cut of the community hierarchy.

    The reports of one cut (``community_depth``), or of a root-down dynamic
    selection (``use_dynamic_selection``), are all mapped in token-sized
    batches; a cut larger than ``max_reports_per_level`` gives way to a
    shallower one, and similarity caps it only when no cut fits.
    """

    community_depth: int = Field(
        default=2,
        ge=0,
        description=(
            "The cut of the community hierarchy whose reports global search "
            "maps over, counted from the root as MS GraphRAG's community_level "
            "(default 2 there too): the communities at this depth plus the "
            "leaves above it, so every entity is covered once. 0 = the root "
            "communities (fewest, broadest reports); a large value = the "
            "leaves. With use_dynamic_selection it bounds only the depths a "
            "selection that found nothing relevant rates whole (MS GraphRAG's "
            "max_level); relevant reports are followed to the leaves."
        ),
    )
    max_reports_per_level: int = Field(
        default=100,
        ge=1,
        description=(
            "Upper bound on the reports one cut (or one step of the dynamic "
            "selection) contributes. A cut with more reports gives way to the "
            "deepest shallower cut that fits (global_cut_depth in the search "
            "metadata); when even the root cut does not fit, or under a filter "
            "on a report key (community_id, level, rating), to the reports "
            "of the cut most similar to the query (a hybrid search restricted "
            "to the cut, global_level_capped). "
            "At most indexing.opensearch.max_query_size - 1 (and the reports "
            "index's live max_result_window - 1): one store request returns at "
            "most that many hits and the cut asks for one more than the cap "
            "(the default of 100 maps at most 99). MS GraphRAG has no cap: it "
            "maps the whole cut. Each report costs map-step input tokens."
        ),
    )
    use_dynamic_selection: bool = Field(
        default=False,
        description=(
            "MS GraphRAG dynamic community selection instead of a fixed cut: "
            "rate the root reports for the query with "
            "models.roles.community_relevance, descend into the children of "
            "the relevant ones down to the leaves (dropping a parent once a "
            "child is relevant), and rate the whole next depth when a depth "
            "yields none, down to community_depth (MS GraphRAG's selection "
            "stops at its community_level; this one follows relevant reports "
            "to the leaves). One fast-model call per rated report, "
            "search.max_concurrency at a time; the relevant reports are "
            "then mapped. Off by default, as in MS GraphRAG."
        ),
    )
    relevance_threshold: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description=(
            "Dynamic selection keeps a report whose relevance (MS GraphRAG's "
            "0-5 rating divided by 5) is at least this. 0.2 = a rating of 1 or "
            "more, MS GraphRAG's threshold: a pruned report prunes its whole "
            "subtree."
        ),
    )
    enable_map_reduce: bool = Field(
        default=True,
        description="Map the selected reports into scored key points. false = "
        "the reports go to the answer model as they are (budget-trimmed).",
    )
    map_reduce_min_results: int = Field(
        default=3,
        ge=1,
        description="Minimum community results before map-reduce synthesis is "
        "applied; below this the results are returned directly.",
    )
    map_context_tokens: int = Field(
        default=12000,
        ge=128,
        description=(
            "Report tokens packed into one map-step LLM call (MS GraphRAG's "
            "data_max_tokens, 12000). Reports are packed in the order the "
            "selection gives them (rank for the fixed cut, rating for dynamic "
            "selection, similarity for the similarity fallback; MS GraphRAG "
            "shuffles them first) until the next one would exceed it; "
            "a longer report gets a call of its own. "
            "Map calls run search.max_concurrency at a time."
        ),
    )
    map_relevance_threshold: int = Field(
        default=0,
        ge=0,
        le=100,
        description="Drop map-step key points whose 0-100 relevance score is at "
        "or below this threshold before ranking (0 keeps any positive score).",
    )
    max_map_reduce_tokens: int = Field(
        default=8000,
        ge=128,
        description="Token budget for the ranked key points handed to the "
        "answer model; the highest-scored points are taken until this budget is "
        "reached, so the answer is written from focused, ranked evidence. "
        "The reports the points cite follow them in the context.",
    )
    text_unit_slots: int = Field(
        default=0,
        ge=0,
        description=(
            "Text units of the communities behind the key points added to the "
            "context (retrieved for the query and reranked). 0 (default) = "
            "none: MS GraphRAG's global context is community reports only, and "
            "chunks next to them took most of the answer budget the reports "
            "and key points need. Local and mix search cover chunk evidence."
        ),
    )
    evidence_tokens: int | None = Field(
        default=4000,
        ge=0,
        description=(
            "Tokens of the cited reports' text that follow the key points in "
            "the answer context: whole reports, best-scored point first, "
            "until the next would exceed it. 4000 (default): measured on 50 "
            "global questions over a 3,318-document MuSiQue corpus, it cut "
            "the cost per global query by 41% ($0.1016 -> $0.0597) against "
            "no cap, with no consistent quality difference across three "
            "pairwise judges (position consistency 0.43-0.57), and the answer "
            "keeps its sources. null = no cap (the default before): as many "
            "as the answer's context budget holds, which on a large cut is "
            "most of a global query's input tokens. 0 = the key points only, "
            "as MS GraphRAG's reduce step reads them (-49% cost, also no "
            "measured quality difference); the answer then reports no "
            "sources, since the key points are model output. Applies to the "
            "map-reduce path (enable_map_reduce)."
        ),
    )


class TypeQuotaConfig(BaseModel):
    """Fusion slots reserved per section type, ``max(multiplier * top_k,
    floor)``, in local search and the strategies built on it (DRIFT's
    follow-up local searches, ppr, iterative's rounds).

    Without them one type (usually entities, which arrive with many near-tie
    fusion scores) takes the whole flat top_k cut and crowds out the chunks
    that carry the answer; MS GraphRAG's local search assembles per-section
    budgets the same way (top_k_entities and top_k_relationships 10). What
    the context then holds of each type is search.token_manager.type_budgets.
    Only the entity and relationship floors follow upstream; the text,
    community and claim values are unmeasured choices (no A/B in
    docs/concepts/defaults-and-evidence.md).
    """

    text_multiplier: float = Field(
        default=2.0,
        ge=0.0,
        description="Text-unit (chunk) slots as a multiple of top_k. "
        "Unmeasured choice.",
    )
    text_floor: int = Field(
        default=20,
        ge=1,
        description="Minimum text-unit slots whatever top_k. Unmeasured choice.",
    )
    entity_multiplier: float = Field(
        default=1.0,
        ge=0.0,
        description="Entity slots as a multiple of top_k. Unmeasured choice.",
    )
    entity_floor: int = Field(
        default=10,
        ge=1,
        description="Minimum entity slots (MS GraphRAG local's top_k_entities).",
    )
    relationship_multiplier: float = Field(
        default=1.0,
        ge=0.0,
        description="Relationship slots as a multiple of top_k (also caps local "
        "search's bridge relationships). Unmeasured choice.",
    )
    relationship_floor: int = Field(
        default=10,
        ge=1,
        description="Minimum relationship slots (MS GraphRAG local's "
        "top_k_relationships).",
    )
    community_multiplier: float = Field(
        default=0.5,
        ge=0.0,
        description="Community report slots as a multiple of top_k. "
        "Unmeasured choice.",
    )
    community_floor: int = Field(
        default=2,
        ge=1,
        description="Minimum community report slots. Unmeasured choice.",
    )
    claim_multiplier: float = Field(
        default=0.5,
        ge=0.0,
        description="Claim slots as a multiple of top_k. Unmeasured choice.",
    )
    claim_floor: int = Field(
        default=2,
        ge=1,
        description="Minimum claim slots. Unmeasured choice.",
    )

    def slots(self, top_k: int) -> dict[str, int]:
        """Reserved slots per section type (its value, ``"text"``) for
        ``top_k``."""
        return {
            kind: max(
                int(getattr(self, f"{kind}_multiplier") * top_k),
                getattr(self, f"{kind}_floor"),
            )
            for kind in ("text", "entity", "relationship", "community", "claim")
        }


class LocalSearchConfig(BaseModel):
    entity_frequency_threshold: int = Field(
        default=20,
        ge=1,
        description="Drop graph-expanded entities appearing in more than this "
        "many text units (too generic to be discriminative for local search) "
        "from the context and the text-unit fan-out. The query's own matched "
        "(seed) entities are always kept.",
    )
    include_bridge_relationships: bool = Field(
        default=True,
        description=(
            "Also fetch the relationships incident to the graph-expanded "
            "entities, edges between two of those entities first (MS GraphRAG "
            "local's in-network relationships). These bridge edges carry the "
            "hops of a multi-hop chain, which the relationship vector query "
            "alone often misses. Needs "
            "indexing.opensearch.build_relationship_vector_index. false = only"
            " the vector query on the relationship index."
        ),
    )
    seed_score_ratio: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description=(
            "Each query entity is looked up in the entities index on its own "
            "and seeds graph expansion with the entities of its name; with "
            "none, with its top hit only when that scores at least this "
            "fraction of the best hit of any query entity (and is no one-word "
            "lookalike: a name sharing one word with it, neither holding the "
            "other), so a generic word "
            "the entity extraction returned seeds no unrelated entity. The "
            "fraction is read at equal hybrid weights, where a hit only the "
            "vector search found scores at most 0.5 of one both found, so "
            "above 0.5 it drops such a hit when another query entity matched "
            "by name; at other weights it is rescaled so that it still does "
            "(0.5 maps to search.hybrid.vector_weight: 0.6 is 0.84 at a 0.8 "
            "vector weight). A vector or lexical search compares raw scores, "
            "where the filter is weaker (similarities of related and "
            "unrelated entities are close). 0 keeps every query entity's top "
            "hit. The seeds share the expansion budget in proportion to their "
            "scores."
        ),
    )
    report_selection: Literal["membership", "similarity"] = Field(
        default="similarity",
        description=(
            "How local search (and each DRIFT follow-up) picks its community "
            "reports. 'membership' (MS GraphRAG local's, on leaf communities "
            "where upstream takes each entity's community at its "
            "community_level cut): the reports of the level-0 (leaf) "
            "communities the graph-expanded entities belong to, most "
            "matched entities first, then by report rank; when no expanded "
            "entity names a community (under a document filter, whose walk "
            "reads the entities index, or when graph expansion failed), or "
            "when a community_id or level filter admits none of those "
            "communities, the 'similarity' reports instead. Search metadata "
            "records which one ran (report_selection; DRIFT counts its "
            "follow-ups by each). 'similarity': the reports most "
            "similar to the joined query entities, which always fill the "
            "report slots. The default: on 200 MuSiQue questions (two "
            "repeats), membership scored exact match 4.0 points (95% CI "
            "-7.75 to -0.25) and token F1 4.1 points (-7.7 to -0.6) lower, "
            "for about 0.5% fewer tokens."
        ),
    )
    text_unit_ranking: Literal["support", "ppr"] = Field(
        default="ppr",
        description=(
            "How local search orders the text units of its expanded entities "
            "before the text quota cuts them (DRIFT's follow-up local searches: "
            "search.drift_search.text_unit_ranking). 'ppr' "
            "(the default): HippoRAG 2's Personalized PageRank over the "
            "expanded entities, their relationships and their text units, "
            "seeded by the seed entities (hit score over the text units "
            "cited), the endpoints of the question's ppr_seed_relationships "
            "relationship hits (kNN) and its "
            "search.ppr_search.passage_seed_top_k text-unit hits (BM25 + "
            "vector, weighted by passage_node_weight), which it ranks too; the "
            "PageRank settings are search.ppr_search's. 'support': by how many "
            "expanded entities cite a unit, then the best citing entity's "
            "score. ppr is the default because it answered multi-hop "
            "questions better than support (the measurement: "
            "docs/concepts/defaults-and-evidence.md#local-ppr-ranking). Costs a text-unit "
            "search, a relationship search, an edge read and a query "
            "embedding a query. See docs/design.md §6.4."
        ),
    )
    ppr_seed_relationships: int = Field(
        default=5,
        ge=0,
        description=(
            "text_unit_ranking: ppr only: also seed the PageRank with the "
            "endpoints of the question's this many relationships-index hits "
            "(HippoRAG 2's query-to-fact seeding, under "
            "search.ppr_search.fact_search_type), each adding the mean score "
            "of its hits to the entity's own seed score before the division by "
            "the text units it cites; an endpoint the expansion did not reach "
            "is read from the entities index. 0 = off (the measured setting "
            "is 5). Needs indexing.opensearch.build_relationship_vector_index."
        ),
    )
    entity_candidates_per_focus: int = Field(
        default=2,
        ge=1,
        description=(
            "Entities-index hits fetched per query entity, among which its "
            "exact name match is looked for (else its top hit seeds, under "
            "seed_score_ratio). Unmeasured choice (no A/B in "
            "docs/concepts/defaults-and-evidence.md)."
        ),
    )
    type_quota: TypeQuotaConfig = Field(
        default_factory=TypeQuotaConfig,
        description=(
            "Fusion slots reserved per section type, max(multiplier * top_k, "
            "floor), in local search and the strategies built on it (DRIFT's "
            "follow-ups, ppr, iterative's rounds); "
            "search.token_manager.type_budgets sets what the context then "
            "holds of each. Only the entity and relationship floors follow MS "
            "GraphRAG; the rest are unmeasured choices."
        ),
    )


class DriftSearchConfig(BaseModel):
    """MS GraphRAG DRIFT search: a community-report primer, then local search.

    The primer expands the query into a hypothetical answer (HyDE), reads the
    ``k_followups`` community reports most similar to it in ``primer_folds``
    calls and drafts intermediate answers plus follow-up queries. Each of
    ``n_depth`` rounds then runs the ``k_followups`` best-placed open follow-up
    queries as local searches, each answered by ``models.roles.drift_action`` with its
    own follow-up queries. The answer model reduces the ranked intermediate
    answers with the local evidence they came from, reranked once.

    The action local searches pack their context by
    ``search.token_manager.type_budgets`` (text 0.5), as every local search
    does, not by MS GraphRAG's DRIFT-specific ``local_search_text_unit_prop``
    0.9 / ``local_search_community_prop`` 0.1.
    """

    k_followups: int = Field(
        default=10,
        ge=1,
        description=(
            "Community reports the primer reads, and follow-up queries run per"
            " round (MS GraphRAG drift_k_followups: 20). Each follow-up query "
            "is one local search plus one models.roles.drift_action call."
        ),
    )
    primer_folds: int = Field(
        default=5,
        ge=1,
        description="Primer calls the primer reports are split over (MS "
        "GraphRAG primer_folds, 5); each drafts its own intermediate answer and "
        "follow-up queries.",
    )
    n_depth: int = Field(
        default=2,
        ge=1,
        description="Rounds of follow-up queries (MS GraphRAG n_depth: 3). Up to "
        "k_followups * n_depth local searches per query, so the defaults make "
        "at most 1 + primer_folds + 20 + 1 = 27 LLM calls (MS GraphRAG's "
        "defaults: 67).",
    )
    local_context_tokens: int = Field(
        default=12000,
        ge=128,
        description="Token budget of the local-search context one follow-up "
        "query is answered from (MS GraphRAG local_search_max_data_tokens).",
    )
    max_answer_tokens: int = Field(
        default=8000,
        ge=128,
        description="Token budget for the ranked intermediate answers handed to "
        "the answer model; the best-scored answers are taken first.",
    )
    text_unit_ranking: Literal["support", "ppr"] = Field(
        default="support",
        description=(
            "How each follow-up local search orders its text units "
            "(search.local_search.text_unit_ranking's values). support (the "
            "default): what DRIFT was measured with. ppr was measured for "
            "local alone and costs each follow-up a text-unit search, a "
            "relationship search, an edge read and a query embedding (up to "
            "k_followups * n_depth = 20 of each a query, near the query "
            "deadline), so DRIFT keeps support until a run measures it."
        ),
    )


class LightRAGSearchConfig(BaseModel):
    raw_query_fallback_max_len: int = Field(
        default=50,
        ge=0,
        description=(
            "When dual-level keyword extraction yields no keywords, a query "
            "shorter than this many characters (in every script, as upstream "
            "LightRAG's len(query) < 50) is searched as its own low-level "
            "keyword. A longer one skips the fallback, as upstream does, since "
            "a whole question as one keyword scans the entity index for a "
            "sentence, not a name: hybrid then retrieves nothing and mix only "
            "chunks. 0 turns the fallback off."
        ),
    )
    kg_stream_top_k: int = Field(
        default=40,
        ge=1,
        description=(
            "Width of the entity and relationship vector queries (LightRAG's "
            "`QueryParam.top_k`, default 40). Separate from the chunk width: "
            "collapsing both onto the caller's single top_k starves the graph "
            "streams. Acts as a floor — a caller asking for a larger top_k gets it."
        ),
    )
    chunk_stream_top_k: int = Field(
        default=20,
        ge=1,
        description=(
            "Width of the chunk stream, and the cap on the merged chunk pool "
            "(LightRAG's `QueryParam.chunk_top_k`, default 20). Acts as a floor "
            "against the caller's top_k."
        ),
    )
    related_chunk_number: int = Field(
        default=5,
        ge=1,
        description=(
            "Chunks allocated per matched entity/relationship when following "
            "`text_unit_ids` lineage (LightRAG's `related_chunk_number`). Slots are "
            "distributed on a decreasing gradient so the last match still gets one."
        ),
    )
    kg_chunk_pick_method: Literal["vector", "weight"] = Field(
        default="weight",
        description=(
            "How hybrid/mix pick the chunks the matched entities and "
            "relationships cite (LightRAG's `kg_chunk_pick_method`). weight: "
            "related_chunk_number chunks per hit on a decreasing gradient, "
            "those more matched hits cite first, then by chunk id (upstream's "
            "order), fetched by id. vector: per side (entities, then "
            "relationships), the cited chunks the query matches best, by one "
            "search of the text units restricted to them (the request's "
            "search_type: BM25 + vector by default), "
            "related_chunk_number * hits / 2 of them up to chunk_stream_top_k. "
            "Upstream defaults to vector; weight stays the default here: on "
            "200 MuSiQue questions vector measured no significant difference "
            "(answer_contains mix -1.5 points [-4.5, +1.25], hybrid -1.25 "
            "[-4.5, +2.0]) and about 1.5 s more latency per query."
        ),
    )
    enable_graph_expansion: bool = Field(
        default=False,
        description=(
            "hybrid/mix: also expand the entity hits and relationship endpoints "
            "through the graph (search.graph_expansion.max_hops) and add the "
            "neighbourhood as entity candidates. Off by default: upstream LightRAG "
            "has no multi-hop traversal (its graph context is the matched items "
            "plus their one-hop incident edges and endpoint entities, which this "
            "strategy already retrieves), and the extra Neptune round trip adds "
            "latency and entities less related to the query."
        ),
    )


class IterativeSearchConfig(BaseModel):
    """Iterative bridge follow-up search (``iterative``; ``auto`` routes
    multi-hop questions to it, no default selects it).

    Round 0 is the ``base`` strategy's search. A fast-model controller
    (``models.roles.query_controller``) reads the question and the packed
    context and says whether it answers the question; when it does not, each
    follow-up round reruns the base search seeded with the bridge entities
    it names and searches the text units for its follow-up queries, fuses
    the new candidates with the earlier rounds' (RRF), reranks them against
    the question and the follow-up queries and repacks them under the same
    token budget. The answer model answers once, from the final context.
    """

    base: Literal["local", "mix"] = Field(
        default="mix",
        description=(
            "The strategy round 0 runs and each follow-up round reruns, seeded "
            "with the bridge entities: local (its query entities) or mix (its "
            "low-level keywords). iterative reads the query extraction, the "
            "indexes and the context packing of this strategy."
        ),
    )
    max_rounds: int = Field(
        default=2,
        ge=1,
        le=5,
        description=(
            "Follow-up rounds after round 0 at most. The controller judges the "
            "context after every round, the last one included, so a query "
            "makes at most max_rounds + 1 controller calls and max_rounds + 1 "
            "base searches; the answer model is called once."
        ),
    )
    round_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description=(
            "Text units each follow-up query retrieves by the request's search_type (BM25 + vector by "
            "default). The base rerun of a follow-up round keeps the query's "
            "top_k."
        ),
    )
    max_next_queries: int = Field(
        default=2,
        ge=0,
        le=5,
        description=(
            "Follow-up queries the controller may ask for in one round, each "
            "naming a bridge entity; a query already searched is not searched "
            "again. 0 follows the bridge entities alone."
        ),
    )
    max_bridge_entities: int = Field(
        default=5,
        ge=0,
        description=(
            "Bridge entities one round follows at most (the controller's "
            "bridge_entities beyond them are dropped, so a controller that "
            "lists many does not multiply the round's lookups); one already "
            "followed is not followed again. 0 follows the follow-up queries "
            "alone. Unmeasured choice (no A/B in "
            "docs/concepts/defaults-and-evidence.md)."
        ),
    )
    controller_context_tokens: int | None = Field(
        default=None,
        ge=256,
        description=(
            "Token budget of the context each controller call reads: the round's candidates packed as the "
            "answer's context is, cut to this. null (the default) packs them "
            "to the answer's budget (the request's max_tokens, else "
            "search.token_manager.max_context_tokens), so the controller "
            "judges the context the answer reads; a smaller value saves "
            "fast-model input tokens, but with large chunks the controller "
            "then reads only the first of the passages the answer gets. The "
            "answer itself always gets its full budget."
        ),
    )


class PPRSearchConfig(BaseModel):
    """HippoRAG 2 Personalized PageRank (the ``ppr`` strategy; its PageRank
    settings also serve search.local_search.text_unit_ranking: ppr).

    The ppr strategy is opt-in (auto does not route to it): the defaults are
    HippoRAG 2's (damping 0.5, passage node weight 0.05, link_top_k 5), and
    docs/concepts/defaults-and-evidence.md#the-ppr-strategy has the
    measurement of its answers. local's default
    ranking uses the PageRank ones (damping, passage node weight, passage
    seeds, iterations, edges per entity) but seeds by its own
    search.local_search.ppr_seed_relationships with no link_top_k cut. See
    docs/design.md §6.4.
    """

    fact_top_k: int = Field(
        default=5,
        ge=1,
        description=(
            "Relationships-index hits of the question (the facts, searched by "
            "fact_search_type) that seed the ppr strategy's PageRank through "
            "their endpoint entities, each endpoint weighted by the mean score "
            "of its facts (relative to the best) over the text units it cites "
            "(HippoRAG 2's linking_top_k, 5); the facts also form the "
            "ppr_facts fusion bucket. Without the relationship vector index "
            "the entities-index hits of the question seed instead, this many."
        ),
    )
    fact_search_type: SearchType | None = Field(
        default=SearchType.VECTOR,
        description=(
            "How the facts (relationships-index hits) are searched, for the "
            "ppr strategy and local's ppr_seed_relationships: vector (the "
            "default: kNN of the question against the relationship "
            "descriptions, HippoRAG 2's query-to-fact similarity and the "
            "measured setting), lexical, hybrid, or null for the request's "
            "search_type. The passage seeds always use the request's (BM25 + "
            "vector by default)."
        ),
    )
    link_top_k: int = Field(
        default=5,
        ge=1,
        description=(
            "Seed entities that keep their weight, best first (HippoRAG 2's "
            "link_top_k)."
        ),
    )
    entity_seed_top_k: int = Field(
        default=0,
        ge=0,
        description=(
            "Also seed with this many entities-index hits of the question "
            "(weighted by hit score over the text units they cite). 0 = off "
            "(HippoRAG 2 seeds from facts only)."
        ),
    )
    passage_seed_top_k: int = Field(
        default=20,
        ge=0,
        description=(
            "Text-unit hits of the question (BM25 + vector under the default "
            "hybrid search) that seed the PageRank as passage nodes, weighted "
            "by passage_node_weight times their min-max normalized score; "
            "in the ppr strategy they also form the ppr_seed_text_units fusion "
            "bucket. 0 = no passage seeds."
        ),
    )
    passage_node_weight: float = Field(
        default=0.05,
        ge=0.0,
        description=(
            "Reset weight of the best passage seed relative to a fact's score "
            "(HippoRAG 2's passage_node_weight, 0.05): small, so the facts' "
            "entities steer the walk and the passage search breaks ties."
        ),
    )
    damping: float = Field(
        default=0.5,
        gt=0.0,
        lt=1.0,
        description=(
            "Probability that the PageRank walk follows an edge rather than "
            "restarting at the seeds (HippoRAG 2: 0.5). At 0.5 at most "
            "0.5^(h+1) of the mass lies more than h hops from the seeds, "
            "which is why a bounded neighbourhood stands in for the whole "
            "graph."
        ),
    )
    max_iterations: int = Field(
        default=30,
        ge=1,
        description="Power-iteration steps at most",
    )
    tolerance: float = Field(
        default=1e-6,
        gt=0.0,
        description="The power iteration stops once the L1 change is below this",
    )
    max_hops: int = Field(
        default=2,
        ge=1,
        description=(
            "Depth of the ppr strategy's graph expansion around the seed "
            "entities (local's ranking uses local's own expansion)."
        ),
    )
    max_neighbourhood_entities: int = Field(
        default=100,
        ge=1,
        description=(
            "Entities besides the seeds the ppr strategy's expansion keeps "
            "(its top_k; the expansion over-fetches by "
            "search.graph_expansion.traversal_fetch_multiplier and splits the "
            "budget across the seeds by fact score)."
        ),
    )
    max_edges_per_vertex: int = Field(
        default=50,
        ge=1,
        description=(
            "Relationships one entity contributes to the PageRank graph at "
            "most, the first by relationship id, so a hub cannot fill it."
        ),
    )
    entity_frequency_threshold: int = Field(
        default=20,
        ge=1,
        description=(
            "An entity citing more document text units than this links to "
            "none of them in the ppr strategy's graph, a seed too (it still "
            "passes its mass on through its relationships); local's ranking "
            "uses search.local_search.entity_frequency_threshold."
        ),
    )
    rerank: bool = Field(
        default=True,
        description=(
            "Rerank the ppr strategy's passages with the configured reranker, "
            "as local reranks its text units; false keeps the fused order "
            "(rerank status skipped), closer to HippoRAG 2, which does not "
            "rerank passages."
        ),
    )


class TreeSearchConfig(BaseModel):
    """RAPTOR summary-tree search (``tree``, opt-in; the
    ``raptor`` methodology).

    Fuses, by RRF, a hybrid search of the text units, one of every level of
    the summary tree (RAPTOR's collapsed tree) and, where the namespace has
    an entities index, the summaries and text units that cite the entities a
    hybrid search with the raw query finds; reranks the leaves and the
    summaries together; keeps at most ``max_nodes_per_level`` summaries of a
    level and expands each to ``expand_children`` of its own passages, so
    the answer cites the original text.
    """

    graph_signal: Literal["auto", "off"] = Field(
        default="auto",
        description="auto: add the graph buckets (tree_graph_summaries, "
        "tree_graph_text_units) where the namespace's methodologies build the "
        "entities index (graphrag or lightrag besides raptor); off: never. "
        "No model call either way.",
    )
    graph_seed_entities: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Entities the raw-query entity search takes as graph seeds.",
    )
    text_top_k: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Text units each text bucket searches for, and the most "
        "the fused result keeps (expanded passages aside).",
    )
    summary_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Summaries the summary buckets search for, and the most "
        "the fused result keeps (RAPTOR's collapsed tree takes top_k 10).",
    )
    max_nodes_per_level: int = Field(
        default=3,
        ge=1,
        le=50,
        description="Summaries of one tree level the context keeps at most, so "
        "neither the roots nor the first level take every summary slot.",
    )
    expand_children: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Passages of its own lineage each kept summary brings "
        "(the best by the query, under the caller's filters), so the answer "
        "cites original text. 0 keeps the summaries alone (RAPTOR's "
        "collapsed tree).",
    )


def _structured_fallback(value: SearchStrategy | None) -> SearchStrategy | None:
    if value in (SearchStrategy.AUTO, SearchStrategy.STRUCTURED):
        raise ValueError(
            f"fallback_strategy cannot be '{value.value}': name the strategy a "
            "query without a structured answer runs (local by default), or "
            "null to answer it from no context"
        )
    return value


class StructuredSearchConfig(BaseModel):
    """Structured graph queries (the ``structured`` strategy, opt-in; docs/
    design.md §6.6, guide docs/guides/structured-queries.md).

    A model writes the question as a JSON plan over the graph's schema, never
    query text; code validates it, compiles it to Gremlin or SQL scoped to
    the query's suffix, checks the compiled form and runs it read-only. A
    question the plan cannot answer runs ``fallback_strategy``. Not routable
    by auto by default (add it to search.auto_routable_strategies once the
    structural benchmark shows a gain).
    """

    mode: Literal["fallback", "augment"] = Field(
        default="fallback",
        description=(
            "fallback: answer from the graph query result, and run "
            "fallback_strategy only when no structured answer is produced "
            "(no valid plan, unanswerable, an unlinked name, an empty result, "
            "a store refusal or time limit; metadata.structured.reason says "
            "which). augment: run fallback_strategy on every query beside the "
            "graph query and fuse both contexts, the query result first: more "
            "recall on mixed questions at the fallback's cost on every query."
        ),
    )
    fallback_strategy: Annotated[
        SearchStrategy | None, AfterValidator(_structured_fallback)
    ] = Field(
        default=SearchStrategy.LOCAL,
        description=(
            "The strategy a query without a structured answer runs (and the "
            "one augment runs beside it), reading the plan's entity names as "
            "its query entities or keywords, so no extraction call is added. "
            "One the namespace does not serve gives way to local, then mix. "
            "null answers such a query from no context. auto and structured "
            "are refused."
        ),
    )
    max_hops: int = Field(
        default=3,
        ge=1,
        le=4,
        description="The longest walk a plan may ask for, in hops over all its edges.",
    )
    rows_in_context: int = Field(
        default=50,
        ge=1,
        le=200,
        description=(
            "Rows, groups or paths written into the query result's context "
            "section, and entities given their own section; for an answer "
            "that names none (a count, an aggregate), the first matches by "
            "rank read as its evidence."
        ),
    )
    max_scan_rows: int = Field(
        default=5000,
        ge=1,
        le=100_000,
        description=(
            "Distinct matches a store returns to code at most: the rows that "
            "code orders by rank, and, under a document filter or a "
            "seed-property condition (which the graph cannot evaluate), every "
            "match it checks and aggregates. Past it the answer is marked "
            "truncated (a count reads 'at least N'). A count without either "
            "is computed by the store and exact."
        ),
    )
    store_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        description=(
            "Server-side time limit of one structured query (Gremlin "
            "evaluationTimeout, the SQLite progress handler), capped at "
            "search.store_timeout_seconds; past it the query falls back "
            "(reason timeout). Neptune's own neptune_query_timeout still "
            "applies (StrictTimeoutValidation refuses a longer one)."
        ),
    )
    repair_attempts: int = Field(
        default=1,
        ge=0,
        le=3,
        description=(
            "Planner calls after one whose plan was refused, each shown the "
            "refusal (the plan field and why)."
        ),
    )
    schema_source: Literal["both", "declared", "observed"] = Field(
        default="both",
        description=(
            "What the planner's schema lists: the extraction schema "
            "(processing.graph_extraction, inline) and what the graph "
            "holds (both), the declared schema alone, or the graph's "
            "observed profile alone. declared reads no profile from the "
            "store; a schema_source schema reaches the planner through the "
            "types it stored."
        ),
    )
    max_schema_tokens: int = Field(
        default=3000,
        ge=200,
        description=(
            "Budget of the planner's schema text, in the answer model's "
            "tokens; the least frequent patterns, properties and types are "
            "cut first."
        ),
    )
    profile_sample_vertices: int = Field(
        default=2000,
        ge=1,
        le=100_000,
        description=(
            "Entities read to build the observed profile (one bounded read "
            "per suffix and profile_ttl_seconds), so a large graph costs no "
            "scan."
        ),
    )
    profile_edges_per_vertex: int = Field(
        default=20,
        ge=0,
        le=1000,
        description="Out-edges of each sampled entity the profile reads at most.",
    )
    profile_ttl_seconds: float = Field(
        default=600.0,
        ge=0,
        description="Seconds a strategy reuses a suffix's profile (0: read it on every query).",
    )
    max_example_values: int = Field(
        default=10,
        ge=0,
        le=50,
        description=(
            "Sampled values shown for a string property with at most this "
            "many distinct short values, so a condition uses the stored "
            "spelling (a country, a status); 0 shows none. Entity names are "
            "never listed: names in the question are linked by code."
        ),
    )
    link_candidates: int = Field(
        default=3,
        ge=1,
        le=20,
        description=(
            "Entities a name in the question links to at most when no entity "
            "has its name (by compact_name_key) among a hybrid entity search's "
            "hits: the best hits of a lexical (BM25) search for it whose names "
            "hold every word of it (legal forms aside; Han, Hangul and Kana by "
            "character pairs), under the caller's filters and the node's "
            "types. Any other name links nothing (the query falls back): a "
            "vector search returns its nearest entities for any text, a "
            "lexical one every name sharing a word."
        ),
    )
    link_min_score: float = Field(
        default=0.0,
        ge=0.0,
        description="Lexical score a linked entity needs above 0 at least (0: any lexical hit).",
    )
    supporting_chunks: int = Field(
        default=5,
        ge=0,
        le=50,
        description=(
            "Passages fetched for the returned entities: those the entities "
            "cite, fused with a hybrid text search of the question among "
            "them and reranked, so the answer can cite text. 0 fetches none."
        ),
    )


class GraphExpansionConfig(BaseModel):
    """The walk around a query's seed entities, on every graph store
    (Neptune, Gremlin Server, the embedded SQLite graph) and over the
    relationships index under a document filter."""

    model_config = ConfigDict(validate_assignment=True)

    max_hops: int = Field(
        default=3,
        ge=1,
        description=(
            "Depth of the graph expansion around the seed entities at query " "time."
        ),
    )
    max_results_per_hop: int = Field(
        default=50,
        ge=1,
        description="Maximum number of neighbours one node contributes per "
        "traversal hop, so a hub entity cannot fan the expansion out unbounded",
    )
    entity_importance_source: Literal["rank", "degree", "none"] = Field(
        default="rank",
        description=(
            "Graph-expansion relevance orders entities by distance from the "
            "seeds (seeds first) and, at the same distance, by this "
            "importance: 'rank' (the indexed entity rank, its degree), "
            "'degree' (the entity's edges counted at query time, as MS "
            "GraphRAG ranks entities; read as 'rank' under a document filter, "
            "whose walk runs over the search store) or 'none' (a neutral 0.5 "
            "for every entity, so expansion ranks by distance alone)."
        ),
    )
    traversal_fetch_multiplier: int = Field(
        default=3,
        ge=1,
        description=(
            "Graph expansion (the graph walk, or under a document filter the "
            "walk over the relationships index) reaches up to top_k * "
            "retrieval_multiplier * this many entities besides the seeds, "
            "split across the seeds (evenly; for local search's in proportion "
            "to their scores: each seed reaches at most its share), ranks "
            "them by relevance, then keeps every seed plus the "
            "top top_k * retrieval_multiplier of the others. 1 = no "
            "over-fetch: every reached entity is kept, with no cut after "
            "ranking."
        ),
    )
    deterministic_order: bool = Field(
        default=True,
        description=(
            "Neptune graph expansion takes each vertex's max_results_per_hop "
            "neighbours in id order, so two builds of one graph (edges "
            "written in another order) expand to the same entities: "
            "reproducible results across rebuilds. The cost: each expanded "
            "vertex has all its neighbours read and sorted before the cut, "
            "which grows with hub degree (Gremlin Server, 2,500-neighbour "
            "hubs at the defaults: 6.8 to 23.2 ms per query; 60-neighbour "
            "hubs: 4.5 to 4.8 ms; Neptune hub latency is unmeasured). false: "
            "the first neighbours the store returns (an order Neptune does "
            "not specify), so a rebuild can change the results. Of equally "
            "short paths to an entity, the one from the better-ranked seed "
            "is kept, then the first by content, either way. The embedded "
            "SQLite graph store always takes neighbours by id and ignores it."
        ),
    )
