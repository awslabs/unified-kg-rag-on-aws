// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { lazy, Suspense, useEffect, useMemo, useState } from "react";
import { Bars } from "../components/charts";
import { colorSlot } from "../lib/graphColors";

import { Empty, ErrorText, Loading, Notice, PageHead, Pager, Sheet } from "../components/ui";
import { useI18n } from "../i18n/i18n";
import { get } from "../lib/api";
import { navigate } from "../lib/router";
import { useTrail } from "../lib/trail";
import type { GraphData, GraphEntity, GraphEntityRef, GraphNode, NamespaceHeld } from "../lib/types";
import { useApi } from "../lib/useApi";
import { contentLang } from "../lib/contentLang";

// The canvas and its force engine load with the page that draws them.
const GraphCanvas = lazy(() => import("../components/GraphCanvas").then((m) => ({ default: m.GraphCanvas })));

interface GraphFile {
  ref: string;
  label: string;
  /** The namespace the export holds (null: an export of an earlier release),
   * the run that wrote it and the build it shows (its finish time). */
  namespace: string | null;
  run: string;
  built_at: string | null;
  /** A later build of its namespace exists. */
  outdated: boolean;
  modified: number;
  bytes: number;
}

/** Why the config's namespace has a graph or not: ``current`` (shown),
 * ``export_off`` (graph.visualization.export_data off), ``unrecorded`` (no
 * build record) or ``failed`` (building it raised; the log says why). */
interface CurrentGraph {
  state: "current" | "export_off" | "unrecorded" | "failed";
  namespace: string | null;
  built_at: string | null;
}

type Panel = "entity" | "communities" | "analysis";

// Groups the legend names (one series colour each); the rest share "other".
const LEGEND_SLOTS = 8;
const OTHER_GROUPS = "\u0000other";

export function GraphPage({ params }: { params: URLSearchParams }) {
  const { t, date } = useI18n();
  const files = useApi(() => get<{ graphs: GraphFile[]; current: CurrentGraph }>("graphs"), []);
  const [chosen, setChosen] = useState<string | null>(params.get("graph"));
  const ref = chosen ?? files.data?.graphs[0]?.ref ?? null;
  const graph = useApi(() => (ref ? get<GraphData>("graphs/data", { ref }) : Promise.resolve(null)), [ref]);
  const none = !!files.data && !files.data.graphs.length;
  // No build record: an empty namespace, or one indexed before build
  // records were written (the stores answer, yet no graph is current).
  const unrecorded = none && files.data?.current.state === "unrecorded";
  const held = useApi(() => (unrecorded ? get<NamespaceHeld>("namespace/methodologies").catch(() => null) : Promise.resolve(null)), [unrecorded]);
  // The build time (and whether it is outdated) first: a narrow select
  // cuts the end, and a console run's id is long.
  const fileLabel = (g: GraphFile) => {
    const built = g.built_at ? Date.parse(g.built_at) / 1000 : NaN;
    const when = `${date(Number.isNaN(built) ? g.modified : built)}${g.outdated ? ` (${t("graph.outdated")})` : ""}`;
    return [when, g.label, ...(g.run !== g.label ? [g.run] : [])].join(" · ");
  };

  return (
    <>
      <PageHead
        title={t("graph.title")}
        lede={t("graph.lede")}
        aside={
          files.data?.graphs.length ? (
            <label className="field" style={{ minWidth: 240 }}>
              <span className="field-label">{t("graph.file")}</span>
              <select className="select" value={ref ?? ""} onChange={(e) => setChosen(e.target.value)}>
                {files.data.graphs.map((g) => (
                  <option key={g.ref} value={g.ref}>
                    {fileLabel(g)}
                  </option>
                ))}
              </select>
            </label>
          ) : null
        }
      />
      <ErrorText error={files.error ?? graph.error} />
      {none ? <NoGraph state={files.data!.current.state} indexed={held.data?.indexed ?? null} namespace={files.data!.current.namespace} /> : null}
      {/* Building failed with exports to show: the newest is drawn, marked
          outdated in the picker when the namespace was built since. */}
      {!none && files.data?.current.state === "failed" ? <Notice tone="warn" title={t("graph.failedTitle")}>{t("graph.failed")}</Notice> : null}
      {files.loading || (unrecorded && held.loading) || (graph.loading && ref) ? <Loading /> : null}
      {graph.data && ref ? <GraphExplorer key={ref} graphRef={ref} data={graph.data} params={params} /> : null}
    </>
  );
}

/** Why there is no graph to show, and the way to one: Settings when the
 * export is off, Ingest for an empty namespace or one to rebuild once (no
 * build record), nothing more to do when building it failed. */
function NoGraph({ state, indexed, namespace }: { state: CurrentGraph["state"]; indexed: boolean | null; namespace: string | null }) {
  const { t } = useI18n();
  if (state === "unrecorded" && indexed === null) return null;
  const why = state === "export_off" ? "exportOff" : state === "failed" ? "failed" : indexed ? "unrecorded" : "none";
  const action =
    why === "exportOff" ? (
      <button className="button" onClick={() => navigate("settings")}>
        {t("graph.openSettings")}
      </button>
    ) : why === "failed" ? undefined : (
      <button className="button" onClick={() => navigate("ingest")}>
        {t("common.goIngest")}
      </button>
    );
  return (
    <Empty title={t(`graph.${why}Title`, { namespace: namespace ?? "" })} action={action}>
      {t(`graph.${why}`)}
    </Empty>
  );
}

function GraphExplorer({ graphRef, data, params }: { graphRef: string; data: GraphData; params: URLSearchParams }) {
  const { t, number, percent, locale } = useI18n();
  const { trail } = useTrail();
  const [selected, setSelected] = useState<string | null>(params.get("focus"));
  const [community, setCommunity] = useState<string | null>(params.get("community"));
  const [colorBy, setColorBy] = useState<"community" | "type">("community");
  const [panel, setPanel] = useState<Panel>(params.get("community") ? "communities" : "entity");
  const [search, setSearch] = useState("");
  const [active, setActive] = useState(0);
  const edgeTarget = params.get("edge");

  const byId = useMemo(() => new Map(data.nodes.map((n) => [n.id, n])), [data.nodes]);
  // Every entity, drawn or not (search and the table).
  const everyone = useMemo<EntityRow[]>(() => [...data.nodes, ...data.others], [data.nodes, data.others]);
  const nameOf = useMemo(() => new Map(everyone.map((n) => [n.id, n.name])), [everyone]);
  // Past the drawn ones, an entity (and the edges of a drawn one to them)
  // is read from the server.
  const partial = data.others.length > 0;
  const remote = useApi(
    () => (selected && partial ? get<GraphEntity>("graphs/entity", { ref: graphRef, id: selected }) : Promise.resolve(null)),
    [selected, partial, graphRef],
  );
  const neighbours = useMemo(() => {
    const map = new Map<string, Set<string>>();
    for (const e of data.edges) {
      if (!map.has(e.source)) map.set(e.source, new Set());
      if (!map.has(e.target)) map.set(e.target, new Set());
      map.get(e.source)!.add(e.target);
      map.get(e.target)!.add(e.source);
    }
    return map;
  }, [data.edges]);

  // Colours follow the entity's group in a fixed order (largest first), so
  // a selection never repaints the rest. The entities with no community (or
  // no type) are not a group: they take the neutral colour and come last,
  // however many they are.
  const groups = useMemo(() => {
    const ungrouped = colorBy === "community" ? "none" : "unknown";
    const key = (n: GraphNode) => (colorBy === "community" ? (n.community_id ?? ungrouped) : (n.type ?? ungrouped));
    const counts = new Map<string, number>();
    for (const n of data.nodes) counts.set(key(n), (counts.get(key(n)) ?? 0) + 1);
    const order = [...counts.entries()]
      .filter(([k]) => k !== ungrouped)
      .sort((a, b) => b[1] - a[1])
      .map(([k]) => k);
    return { key, order, counts, ungrouped };
  }, [data.nodes, colorBy]);
  const communityById = useMemo(() => new Map(data.communities.map((c) => [c.id, c])), [data.communities]);
  const communityName = (id: string) => {
    if (id === "none") return t("graph.noCommunity");
    const c = communityById.get(id);
    return c ? c.top_entities.slice(0, 2).join(", ") || id.slice(0, 8) : id.slice(0, 8);
  };
  const slotOf = useMemo(() => new Map(groups.order.map((k, i) => [k, i])), [groups.order]);
  const colorOf = (id: string) => {
    const node = byId.get(id);
    return colorSlot(node ? (slotOf.get(groups.key(node)) ?? 99) : 99);
  };
  const groupLabel = (k: string) => `${colorBy === "community" ? communityName(k) : k} (${number(groups.counts.get(k) ?? 0)})`;
  const legend = groups.order.slice(0, LEGEND_SLOTS).map((k, i) => ({ key: k, label: groupLabel(k), color: colorSlot(i) }));
  if (groups.order.length > LEGEND_SLOTS) legend.push({ key: OTHER_GROUPS, label: t("graph.other", { count: groups.order.length - LEGEND_SLOTS }), color: colorSlot(99) });
  if (groups.counts.has(groups.ungrouped)) legend.push({ key: groups.ungrouped, label: groupLabel(groups.ungrouped), color: colorSlot(99) });
  // A legend entry hovered or focused lights its members (a channel that
  // does not rely on colour).
  const [legendGroup, setLegendGroup] = useState<string | null>(null);
  const groupMembers = useMemo(() => {
    if (!legendGroup) return null;
    const inGroup = (n: GraphNode) => {
      const slot = slotOf.get(groups.key(n)) ?? 99;
      return legendGroup === OTHER_GROUPS ? slot >= LEGEND_SLOTS && groups.key(n) !== groups.ungrouped : groups.key(n) === legendGroup;
    };
    return new Set(data.nodes.filter(inGroup).map((n) => n.id));
  }, [legendGroup, data.nodes, groups, slotOf]);

  const highlighted = useMemo(() => {
    if (community) {
      const c = communityById.get(community);
      return c ? new Set(c.nodes) : null;
    }
    if (selected) return new Set([selected, ...(neighbours.get(selected) ?? [])]);
    return null;
  }, [community, selected, neighbours, communityById]);

  const traced = useMemo(() => {
    const ids = new Set<string>();
    if (params.get("focus")) ids.add(params.get("focus")!);
    if (edgeTarget) ids.add(edgeTarget);
    return ids;
  }, [params, edgeTarget]);

  const matches = useMemo(() => rankMatches(everyone, search), [search, everyone]);
  useEffect(() => setActive(0), [search]);

  const fetched = remote.data && remote.data.node.id === selected ? remote.data : null;
  const node = selected ? (fetched?.node ?? byId.get(selected)) : undefined;
  const nodeEdges = !selected ? [] : partial ? (fetched?.edges ?? []) : data.edges.filter((e) => e.source === selected || e.target === selected);
  const neighbourName = (id: string) => nameOf.get(id) ?? fetched?.names[id] ?? id;
  const notDrawn = !!selected && !byId.has(selected);
  const fromTrail = trail && (trail.source.id === params.get("focus") || trail.source.node_ids?.includes(params.get("focus")) || trail.source.community_id === params.get("community"));

  const pick = (id: string) => {
    setSelected(id);
    setCommunity(null);
    setPanel("entity");
    setSearch("");
  };

  return (
    <>
      {fromTrail && trail ? (
        <div className="trail-banner" role="note">
          <span className="dot" aria-hidden="true" />
          <span>{t("graph.trail.question", { question: trail.question.length > 60 ? `${trail.question.slice(0, 59)}…` : trail.question })}</span>
          <span className="seg" aria-hidden="true" />
          <span className="dot" aria-hidden="true" />
          <span>
            #{trail.source.rank} {trail.source.title} ({t(`ask.kind.${trail.source.kind}`)})
          </span>
          {trail.source.supported_by?.length ? (
            <>
              <span className="seg" aria-hidden="true" />
              <span>{t("graph.trail.passages", { count: trail.source.supported_by.length })}</span>
            </>
          ) : null}
          <button className="link-button" style={{ marginLeft: "auto" }} onClick={() => window.history.back()}>
            {t("graph.trail.back")}
          </button>
        </div>
      ) : null}
      <div className="graph-layout">
        <div className="stack" style={{ gap: 10 }}>
          <div className="graph-scene">
            <div className="graph-scene-bar" data-canvas-overlay="">
              <div style={{ position: "relative", flex: "1 1 240px", maxWidth: 420 }}>
                <label className="visually-hidden" htmlFor="graph-search">
                  {t("graph.search")}
                </label>
                <input
                  id="graph-search"
                  className="input"
                  role="combobox"
                  aria-expanded={matches.length > 0}
                  aria-controls="graph-search-results"
                  aria-activedescendant={matches[active] ? `match-${matches[active].id}` : undefined}
                  placeholder={t("graph.searchPlaceholder")}
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "ArrowDown") setActive((a) => Math.min(matches.length - 1, a + 1));
                    else if (e.key === "ArrowUp") setActive((a) => Math.max(0, a - 1));
                    else if (e.key === "Enter" && matches[active]) pick(matches[active].id);
                    else if (e.key === "Escape") setSearch("");
                    else return;
                    e.preventDefault();
                  }}
                />
                {matches.length ? (
                  <div className="search-results" id="graph-search-results" role="listbox">
                    {matches.map((m, i) => (
                      <div key={m.id} id={`match-${m.id}`} role="option" aria-selected={i === active} onMouseDown={(e) => e.preventDefault()} onClick={() => pick(m.id)}>
                        <span>{m.name}</span>
                        <span className="id">
                          {m.type} · {m.degree}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : search.trim() ? (
                  <p className="field-help" role="status">
                    {t("graph.noMatch")}
                  </p>
                ) : null}
              </div>
              <div className="chips" role="group" aria-label={t("graph.colorBy")}>
                <button className="chip" aria-pressed={colorBy === "community"} onClick={() => setColorBy("community")}>
                  {t("graph.byCommunity")}
                </button>
                <button className="chip" aria-pressed={colorBy === "type"} onClick={() => setColorBy("type")}>
                  {t("graph.byType")}
                </button>
              </div>
            </div>
            <Suspense fallback={<Loading />}>
            <GraphCanvas
              data={data}
              colorOf={colorOf}
              selected={selected}
              highlighted={groupMembers ?? highlighted}
              traced={traced}
              tracedEdge={params.get("focus") && edgeTarget ? [params.get("focus")!, edgeTarget] : null}
              onSelect={(id) => {
                setSelected(id);
                setCommunity(null);
                if (id) setPanel("entity");
              }}
              legend={legend}
              onLegendFocus={setLegendGroup}
            />
            </Suspense>
          </div>
          {partial ? <p className="field-help">{t("graph.drawnOnly", { drawn: number(data.nodes.length), total: number(data.stats.nodes) })}</p> : null}
        </div>
        <section className="sheet">
          <div className="tabs" role="tablist" aria-label={t("graph.panels")}>
            {(["entity", "communities", "analysis"] as Panel[]).map((p) => (
              <button key={p} role="tab" id={`tab-${p}`} aria-controls={`panel-${p}`} aria-selected={panel === p} className="tab" onClick={() => setPanel(p)}>
                {t(`graph.panel.${p}`)}
              </button>
            ))}
          </div>
          <div className="sheet-body" role="tabpanel" id={`panel-${panel}`} aria-labelledby={`tab-${panel}`}>
            <h2 className="visually-hidden">{t(`graph.panel.${panel}`)}</h2>
            {panel === "entity" ? <ErrorText error={remote.error} /> : null}
            {panel === "entity" ? (
              selected && partial && !fetched && remote.loading ? (
                <Loading />
              ) : node ? (
                <div className="stack" style={{ gap: 12 }}>
                  <div>
                    <h2 className="sheet-title" style={{ fontSize: "var(--text-lg)" }} lang={contentLang(node.name, locale)}>
                      {node.name}
                    </h2>
                    <div className="row" style={{ marginTop: 4 }}>
                      {node.type ? <span className="badge">{node.type}</span> : null}
                      {node.in_schema === true ? <span className="badge badge-good">{t("graph.inSchema")}</span> : null}
                      {node.in_schema === false ? <span className="badge badge-warn">{t("graph.offSchema")}</span> : null}
                      {node.source_kinds.includes("seed_graph") ? <span className="badge badge-trace">{t("graph.seed")}</span> : null}
                    </div>
                  </div>
                  {notDrawn ? <p className="field-help" style={{ margin: 0 }}>{t("graph.notDrawn")}</p> : null}
                  {node.description ? (
                    <p style={{ margin: 0, whiteSpace: "pre-line", fontSize: "var(--text-sm)" }} lang={contentLang(node.description, locale)}>
                      {node.description}
                    </p>
                  ) : null}
                  <dl className="kv">
                    <dt>{t("graph.degree")}</dt>
                    <dd>{number(node.degree)}</dd>
                    <dt>{t("graph.passages")}</dt>
                    <dd>{number(node.text_unit_ids.length)}</dd>
                    {node.community_id ? (
                      <>
                        <dt>{t("graph.community")}</dt>
                        <dd>
                          <button
                            className="link-button"
                            onClick={() => {
                              setCommunity(node.community_id);
                              setPanel("communities");
                            }}
                          >
                            {communityName(node.community_id)}
                          </button>
                        </dd>
                      </>
                    ) : null}
                    {node.iri ? (
                      <>
                        <dt>IRI</dt>
                        <dd className="id">{node.iri}</dd>
                      </>
                    ) : null}
                    <dt>{t("graph.id")}</dt>
                    <dd className="id">{node.id}</dd>
                  </dl>
                  <div>
                    <h3 className="field-label" style={{ margin: "0 0 6px" }}>
                      {t("graph.neighbours", { count: nodeEdges.length })}
                    </h3>
                    <ul className="list-plain">
                      {nodeEdges.slice(0, 40).map((edge) => {
                        // Told apart by direction and description: two
                        // relationships may join the same pair.
                        const out = edge.source === node.id;
                        const other = out ? edge.target : edge.source;
                        return (
                          <li key={edge.id}>
                            <button onClick={() => pick(other)} title={edge.description ?? undefined}>
                              <span className="neighbour" lang={contentLang(`${neighbourName(other)} ${edge.description ?? ""}`, locale)}>
                                <span>
                                  <span className="edge-dir" aria-hidden="true">
                                    {out ? "→" : "←"}
                                  </span>
                                  <span className="visually-hidden">{t(out ? "graph.edgeOut" : "graph.edgeIn")} </span>
                                  {neighbourName(other)}
                                </span>
                                {edge.description ? <span className="neighbour-desc">{edge.description}</span> : null}
                              </span>
                              <span className="id">{edge.type}</span>
                            </button>
                          </li>
                        );
                      })}
                    </ul>
                  </div>
                </div>
              ) : (
                <div className="stack" style={{ gap: 10 }}>
                  <p className="muted" style={{ margin: 0 }}>
                    {t("graph.pickHint")}
                  </p>
                  <h3 className="field-label" style={{ margin: 0 }}>
                    {t("graph.mostConnected")}
                  </h3>
                  <ul className="list-plain">
                    {data.stats.top_entities.map((e) => (
                      <li key={e.id}>
                        <button onClick={() => pick(e.id)}>
                          <span>{e.name}</span>
                          <span className="id">{e.degree}</span>
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              )
            ) : null}
            {panel === "communities" ? (
              data.communities.length ? (
                <ul className="list-plain">
                  {data.communities.map((c) => (
                    <li key={c.id}>
                      <button aria-pressed={community === c.id} onClick={() => setCommunity(community === c.id ? null : c.id)}>
                        <span>
                          <span className="id">L{c.level}</span> {c.top_entities.slice(0, 3).join(", ") || c.id.slice(0, 8)}
                        </span>
                        <span className="id">{number(c.size)}</span>
                      </button>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="muted">{t("graph.noCommunities")}</p>
              )
            ) : null}
            {panel === "analysis" ? (
              <div className="stack" style={{ gap: 14 }}>
                <dl className="kv">
                  <dt>{t("graph.stats.nodes")}</dt>
                  <dd>{number(data.stats.nodes)}</dd>
                  <dt>{t("graph.stats.edges")}</dt>
                  <dd>{number(data.stats.edges)}</dd>
                  <dt>{t("graph.stats.components")}</dt>
                  <dd>{number(data.stats.components)}</dd>
                  <dt>{t("graph.stats.density")}</dt>
                  <dd>{number(data.stats.density, { maximumFractionDigits: 3 })}</dd>
                  <dt>{t("graph.stats.meanDegree")}</dt>
                  <dd>{number(data.stats.mean_degree, { maximumFractionDigits: 2 })}</dd>
                  <dt>{t("graph.stats.communities")}</dt>
                  <dd>{data.stats.communities_by_level.map(([level, n]) => `L${level}: ${number(n)}`).join(" · ") || "—"}</dd>
                </dl>
                <div>
                  <h3 className="field-label" style={{ margin: "0 0 6px" }}>
                    {t("graph.stats.degreeHistogram")}
                  </h3>
                  <Bars data={data.stats.degree_histogram.map(([d, n]) => ({ key: String(d), value: n }))} format={(v) => number(v)} label={t("graph.stats.degreeHistogram")} />
                </div>
                <CountTable title={t("graph.stats.types")} rows={data.stats.types} total={data.stats.nodes} percent={percent} number={number} />
                <CountTable title={t("graph.stats.relations")} rows={data.stats.relation_types} total={data.stats.edges} percent={percent} number={number} />
                <div>
                  <h3 className="field-label" style={{ margin: "0 0 6px" }}>
                    {t("graph.stats.schema")}
                  </h3>
                  <div className="table-wrap" tabIndex={0}>
                    <table className="table table-compact">
                      <thead>
                        <tr>
                          <th>
                            <span className="visually-hidden">{t("graph.stats.what")}</span>
                          </th>
                          <th className="num">{t("graph.inSchema")}</th>
                          <th className="num">{t("graph.offSchema")}</th>
                          <th className="num">{t("graph.unannotated")}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(["entities", "relationships"] as const).map((k) => (
                          <tr key={k}>
                            <td>{t(`graph.stats.${k}`)}</td>
                            <td className="num">{number(data.stats.schema[k].in_schema)}</td>
                            <td className="num">{number(data.stats.schema[k].off_schema)}</td>
                            <td className="num">{number(data.stats.schema[k].unannotated)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
                <div>
                  <h3 className="field-label" style={{ margin: "0 0 6px" }}>
                    {t("graph.stats.evidence")}
                  </h3>
                  <div className="table-wrap" tabIndex={0}>
                    <table className="table table-compact">
                      <thead>
                        <tr>
                          <th>
                            <span className="visually-hidden">{t("graph.stats.what")}</span>
                          </th>
                          <th className="num">{t("graph.evidence.document")}</th>
                          <th className="num">{t("graph.evidence.seed_graph")}</th>
                          <th className="num">{t("graph.evidence.both")}</th>
                        </tr>
                      </thead>
                      <tbody>
                        {(["entities", "relationships"] as const).map((k) => (
                          <tr key={k}>
                            <td>{t(`graph.stats.${k}`)}</td>
                            <td className="num">{number(data.stats.evidence[k].document ?? 0)}</td>
                            <td className="num">{number(data.stats.evidence[k].seed_graph ?? 0)}</td>
                            <td className="num">{number(data.stats.evidence[k].both ?? 0)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        </section>
      </div>
      <Sheet title={t("graph.nodeTable")} pad={false}>
        <EntityTable nodes={everyone} communityName={communityName} onPick={pick} />
      </Sheet>
    </>
  );
}

/** The entities whose name holds `search` (case-insensitive), best first:
 * the exact name, then names starting with it, then the rest, each by
 * connections. */
export function rankMatches<T extends { name: string; degree: number }>(entities: T[], search: string, limit = 12): T[] {
  const q = search.trim().toLowerCase();
  if (!q) return [];
  const rank = (name: string) => (name === q ? 0 : name.startsWith(q) ? 1 : 2);
  return entities
    .map((n) => ({ n, name: n.name.toLowerCase() }))
    .filter(({ name }) => name.includes(q))
    .sort((a, b) => rank(a.name) - rank(b.name) || b.n.degree - a.n.degree)
    .slice(0, limit)
    .map(({ n }) => n);
}

/** A table row: a drawn entity in full, another by name and degree. */
type EntityRow = GraphEntityRef & Partial<Pick<GraphNode, "type" | "community_id">>;

// Rows the entity table shows at once (a graph can hold tens of thousands).
export const TABLE_PAGE = 100;

/** Every entity, most connected first: built only while open, a page at a
 * time. */
export function EntityTable({ nodes, communityName, onPick }: { nodes: EntityRow[]; communityName: (id: string) => string; onPick: (id: string) => void }) {
  const { t, number } = useI18n();
  const [open, setOpen] = useState(false);
  const [page, setPage] = useState(0);
  const sorted = useMemo(() => (open ? [...nodes].sort((a, b) => b.degree - a.degree) : []), [open, nodes]);
  const shown = sorted.slice(page * TABLE_PAGE, (page + 1) * TABLE_PAGE);
  return (
    <details onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}>
      <summary style={{ padding: "12px 20px" }}>{t("graph.nodeTableShow", { count: nodes.length })}</summary>
      {open ? (
        <>
          <div className="table-wrap" tabIndex={0} style={{ maxHeight: 360 }}>
            <table className="table">
              <thead>
                <tr>
                  <th>{t("graph.name")}</th>
                  <th>{t("graph.type")}</th>
                  <th className="num">{t("graph.degree")}</th>
                  <th>{t("graph.community")}</th>
                </tr>
              </thead>
              <tbody>
                {shown.map((n) => (
                  <tr key={n.id}>
                    <td>
                      <button className="link-button" onClick={() => onPick(n.id)}>
                        {n.name}
                      </button>
                    </td>
                    <td>{n.type ?? "—"}</td>
                    <td className="num">{number(n.degree)}</td>
                    <td>{n.community_id ? communityName(n.community_id) : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Pager page={page} size={TABLE_PAGE} total={nodes.length} onPage={setPage} />
        </>
      ) : null}
    </details>
  );
}

function CountTable({
  title,
  rows,
  total,
  percent,
  number,
}: {
  title: string;
  rows: [string, number][];
  total: number;
  percent: (v: number | null, d?: number) => string;
  number: (v: number | null) => string;
}) {
  return (
    <div>
      <h3 className="field-label" style={{ margin: "0 0 6px" }}>
        {title}
      </h3>
      <table className="table">
        <tbody>
          {rows.slice(0, 10).map(([name, n]) => (
            <tr key={name}>
              <td className="mono">{name}</td>
              <td className="num">{number(n)}</td>
              <td className="num">{percent(total ? n / total : null)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
