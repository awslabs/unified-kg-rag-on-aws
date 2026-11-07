// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/** The shapes the console API returns (application/cli/console/*.py). */

export interface Status {
  version: string;
  config_path: string | null;
  config_error: string | null;
  namespace: string | null;
  backend: Record<string, string | null> | null;
  models: Record<string, string | null> | null;
  methodologies: string[];
  observability: { enabled: boolean; backend: string } | null;
  prices_configured: boolean;
  jobs: JobRecord[];
  loopback: boolean;
  sample_questions: string[];
}

export interface StrategyInfo {
  name: string;
  methodology: string | null;
  routable: boolean;
  available: boolean;
  reason: string | null;
}

export interface Provenance {
  kind: string;
  seed_graphs: string[];
  iri: string | null;
}

export interface Source {
  rank: number;
  kind: "entity" | "relationship" | "community" | "text" | string;
  id: string | null;
  title: string | null;
  score: number | null;
  excerpt: string;
  truncated: boolean;
  document_ids: string[];
  file_name: string | null;
  chunk_id: string | number | null;
  text_unit_ids: string[];
  source_kinds: string[];
  iri: string | null;
  provenance: Provenance | null;
  entity_type?: string | null;
  relation?: string | null;
  node_ids?: (string | null)[];
  edge_id?: string | null;
  community_id?: string | null;
  supported_by?: number[];
  unread_passages?: number;
  /** A graph query result (the structured strategy). */
  operation?: string | null;
  row_count?: number | null;
  exact?: boolean;
  backend?: string | null;
  /** The context marker [n] the answer model saw it under. */
  marker: number | null;
}

export interface CitedDocument {
  document_id: string | null;
  file_name: string | null;
  file_path: string | null;
  page: number | null;
  page_end: number | null;
  text_unit_ids: string[];
}

/** One numbered context section (`RAGOutput.citations`), with the rank of
 * the source it was reported as (null for a summary the model read). */
export interface Citation {
  marker: number;
  cited: boolean;
  section_type: string;
  source_id: string | null;
  title: string | null;
  document_id: string | null;
  file_name: string | null;
  file_path: string | null;
  page: number | null;
  page_end: number | null;
  chunk_id: string | null;
  text_unit_ids: string[];
  documents: CitedDocument[];
  iri: string | null;
  source_rank: number | null;
}

export interface Usage {
  llm_calls: number;
  input_tokens: number;
  output_tokens: number;
  cache_hits: number;
  rerank_calls?: number;
  by_role?: Record<string, Partial<Usage>>;
  by_model?: Record<string, Partial<Usage>>;
}

export interface AskResult {
  answer: string;
  citations: Citation[];
  unknown_citation_markers: number[];
  strategy: string | null;
  routing: { decided_by: "router" | "only_candidate" | "fallback"; candidates: string[] } | null;
  abstained: boolean;
  abstention_reason: string | null;
  blocked: boolean;
  error: boolean;
  error_message: string | null;
  rerank: { status: string; reason?: string; stages?: { scope: string; status: string }[] } | null;
  usage: Usage | null;
  cost_usd: number | null;
  unpriced_models: string[];
  prices_configured: boolean;
  latency_seconds: number;
  total_results: number | null;
  context_sections_included: number | null;
  context_sections_excluded: number | null;
  processed_query: { original_query: string; final_query: string; translated_query: string | null; standalone_query?: string | null; entities: string[] };
  sources: Source[];
  namespace: string;
  trace_id: string | null;
  trace_url: string | null;
}

export type AskEvent =
  | { kind: "start"; query_id: string; trace_id: string | null; trace_url: string | null }
  | { kind: "chunk"; text: string; markers: number[] }
  | ({ kind: "result" } & AskResult)
  | { kind: "error"; error_kind: string; message: string };

export interface StageProgress {
  name: string;
  status: "completed" | "failed" | "running" | "pending" | "restored" | "stopped" | "not_run" | string;
  duration_seconds?: number | null;
  input_count?: number | null;
  output_count?: number | null;
  error?: string | null;
  failed_items?: string[];
  llm_calls?: number | null;
}

export interface JobRecord {
  id: string;
  kind: "ingest" | "evaluate";
  namespace: string;
  params: Record<string, unknown>;
  command: string[];
  env?: Record<string, string>;
  created: number;
  started: number | null;
  ended: number | null;
  status: "running" | "succeeded" | "failed" | "stopping" | "stopped" | "lost";
  exit_code: number | null;
  error: string | null;
  result: Record<string, unknown> | null;
  pipeline_id: string | null;
  output_directory: string | null;
  plan: string[];
  total_runs: number;
  confirmation?: "integrity" | "reset" | "deletions" | null;
  progress?: { stages?: StageProgress[]; done: number; total: number } | null;
  log?: string[];
}

export interface Corpus {
  ref: string;
  name: string;
  files: number;
  unsupported: number;
  bytes: number;
  extensions: Record<string, number>;
  modified: number;
}

export interface GraphNode {
  id: string;
  name: string;
  type: string | null;
  description: string | null;
  community_id: string | null;
  text_unit_ids: string[];
  source_kinds: string[];
  iri: string | null;
  type_iri: string | null;
  in_schema: boolean | null;
  degree: number;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  type: string | null;
  description: string | null;
  weight: number | null;
  in_schema: boolean | null;
  source_kinds: string[];
}

export interface Community {
  id: string;
  level: number;
  parent: string | null;
  size: number;
  nodes: string[];
  top_entities: string[];
}

export interface GraphStats {
  nodes: number;
  edges: number;
  components: number;
  density: number;
  mean_degree: number;
  max_degree: number;
  degree_histogram: [number, number][];
  top_entities: { id: string; name: string; degree: number; type: string | null }[];
  types: [string, number][];
  relation_types: [string, number][];
  communities_by_level: [number, number][];
  schema: Record<"entities" | "relationships", Record<"in_schema" | "off_schema" | "unannotated", number>>;
  evidence: Record<"entities" | "relationships", Record<string, number>>;
}

export interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
  layout: Record<string, [number, number]>;
  communities: Community[];
  stats: GraphStats;
}

export interface MetricStats {
  mean: number;
  queries: number;
  repeat_stdev?: number;
}

export interface Delta {
  pairs: number;
  mean_delta: number;
  ci_low: number;
  ci_high: number;
  wins: number;
  ties: number;
  losses: number;
}

export interface Comparison {
  baseline: string;
  baselines: Record<string, string>;
  strategies: Record<string, { repeats: number; metrics: Record<string, MetricStats>; failed_answers: number; routed_strategies: Record<string, number> }>;
  deltas_vs_baseline: Record<string, Record<string, Delta>>;
  cost?: {
    strategies: Record<string, Record<string, unknown>>;
    deltas_vs_baseline?: Record<string, Record<string, Delta>>;
  };
  warnings?: string[];
  bootstrap?: { confidence?: number; resamples?: number };
  pairwise_vs_baseline?: Record<string, { criteria: Record<string, { candidate_win_rate: number | null }>; lengths?: { buckets: Record<string, { pairs: number; criteria: Record<string, { candidate_win_rate: number | null }> }> } }>;
  closed_book_arms?: string[];
}

export interface RunSummary {
  file: string;
  strategy: string;
  total_queries: number;
  successful_evaluations: number;
  failed_evaluations: number;
  average_response_time: number | null;
  metric_statistics: Record<string, { mean: number; count: number }>;
  abstention_statistics: Record<string, unknown> | null;
  started: string | null;
}

export interface SavedRun {
  label: string;
  summaries: RunSummary[];
  comparison: Comparison | null;
  comparison_file: string | null;
}
