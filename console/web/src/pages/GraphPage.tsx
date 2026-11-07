// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { useEffect, useMemo, useState } from "react";
import { Bars } from "../components/charts";
import { colorSlot, GraphCanvas } from "../components/GraphCanvas";
import { Empty, ErrorText, Loading, PageHead, Sheet } from "../components/ui";
import { useI18n } from "../i18n/i18n";
import { get } from "../lib/api";
import { navigate } from "../lib/router";
import { useTrail } from "../lib/trail";
import type { GraphData, GraphNode } from "../lib/types";
import { useApi } from "../lib/useApi";

interface GraphFile {
  ref: string;
  label: string;
  modified: number;
  bytes: number;
}

type Panel = "entity" | "communities" | "analysis";

export function GraphPage({ params }: { params: URLSearchParams }) {
  const { t, date } = useI18n();
  const files = useApi(() => get<{ graphs: GraphFile[] }>("graphs"), []);
  const [chosen, setChosen] = useState<string | null>(params.get("graph"));
  const ref = chosen ?? files.data?.graphs[0]?.ref ?? null;
  const graph = useApi(() => (ref ? get<GraphData>("graphs/data", { ref }) : Promise.resolve(null)), [ref]);

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
                    {g.label} · {date(g.modified)}
                  </option>
                ))}
              </select>
            </label>
          ) : null
        }
      />
      <ErrorText error={files.error ?? graph.error} />
      {files.data && !files.data.graphs.length ? (
        <Empty
          title={t("graph.noneTitle")}
          action={
            <button className="button" onClick={() => navigate("ingest")}>
              {t("graph.goIngest")}
            </button>
          }
        >
          {t("graph.none")}
        </Empty>
      ) : null}
      {graph.loading && ref ? <Loading /> : null}
      {graph.data ? <GraphExplorer key={ref} data={graph.data} params={params} /> : null}
    </>
  );
}

function GraphExplorer({ data, params }: { data: GraphData; params: URLSearchParams }) {
  const { t, number, percent } = useI18n();
  const { trail } = useTrail();
  const [selected, setSelected] = useState<string | null>(params.get("focus"));
  const [community, setCommunity] = useState<string | null>(params.get("community"));
  const [colorBy, setColorBy] = useState<"community" | "type">("community");
  const [panel, setPanel] = useState<Panel>(params.get("community") ? "communities" : "entity");
  const [search, setSearch] = useState("");
  const [active, setActive] = useState(0);
  const edgeTarget = params.get("edge");

  const byId = useMemo(() => new Map(data.nodes.map((n) => [n.id, n])), [data.nodes]);
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
  // a selection never repaints the rest.
  const groups = useMemo(() => {
    const key = (n: GraphNode) => (colorBy === "community" ? (n.community_id ?? "none") : (n.type ?? "unknown"));
    const counts = new Map<string, number>();
    for (const n of data.nodes) counts.set(key(n), (counts.get(key(n)) ?? 0) + 1);
    const order = [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([k]) => k);
    return { key, order, counts };
  }, [data.nodes, colorBy]);
  const communityName = (id: string) => {
    if (id === "none") return t("graph.noCommunity");
    const c = data.communities.find((x) => x.id === id);
    return c ? c.top_entities.slice(0, 2).join(", ") || id.slice(0, 8) : id.slice(0, 8);
  };
  const colorOf = (id: string) => {
    const node = byId.get(id);
    return colorSlot(node ? groups.order.indexOf(groups.key(node)) : 99);
  };
  const legend = groups.order.slice(0, 8).map((k, i) => ({
    label: `${colorBy === "community" ? communityName(k) : k} (${number(groups.counts.get(k) ?? 0)})`,
    color: colorSlot(i),
  }));
  if (groups.order.length > 8) legend.push({ label: t("graph.other", { count: groups.order.length - 8 }), color: colorSlot(99) });

  const highlighted = useMemo(() => {
    if (community) {
      const c = data.communities.find((x) => x.id === community);
      return c ? new Set(c.nodes) : null;
    }
    if (selected) return new Set([selected, ...(neighbours.get(selected) ?? [])]);
    return null;
  }, [community, selected, neighbours, data.communities]);

  const traced = useMemo(() => {
    const ids = new Set<string>();
    if (params.get("focus")) ids.add(params.get("focus")!);
    if (edgeTarget) ids.add(edgeTarget);
    return ids;
  }, [params, edgeTarget]);

  const matches = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return [];
    return data.nodes
      .filter((n) => n.name.toLowerCase().includes(q))
      .sort((a, b) => b.degree - a.degree)
      .slice(0, 12);
  }, [search, data.nodes]);
  useEffect(() => setActive(0), [search]);

  const node = selected ? byId.get(selected) : undefined;
  const nodeEdges = selected ? data.edges.filter((e) => e.source === selected || e.target === selected) : [];
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
          <div className="row">
            <div style={{ position: "relative", flex: "1 1 260px" }}>
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
          <GraphCanvas
            data={data}
            colorOf={colorOf}
            selected={selected}
            highlighted={highlighted}
            traced={traced}
            tracedEdge={params.get("focus") && edgeTarget ? [params.get("focus")!, edgeTarget] : null}
            onSelect={(id) => {
              setSelected(id);
              setCommunity(null);
              if (id) setPanel("entity");
            }}
            legend={legend}
          />
          {Object.keys(data.layout).length < data.nodes.length ? (
            <p className="field-help">{t("graph.drawnOnly", { drawn: Object.keys(data.layout).length, total: data.nodes.length })}</p>
          ) : null}
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
            {panel === "entity" ? (
              node ? (
                <div className="stack" style={{ gap: 12 }}>
                  <div>
                    <h2 className="sheet-title" style={{ fontSize: "var(--text-lg)" }}>
                      {node.name}
                    </h2>
                    <div className="row" style={{ marginTop: 4 }}>
                      {node.type ? <span className="badge">{node.type}</span> : null}
                      {node.in_schema === true ? <span className="badge badge-good">{t("graph.inSchema")}</span> : null}
                      {node.in_schema === false ? <span className="badge badge-warn">{t("graph.offSchema")}</span> : null}
                      {node.source_kinds.includes("seed_graph") ? <span className="badge badge-trace">{t("graph.seed")}</span> : null}
                    </div>
                  </div>
                  {node.description ? <p style={{ margin: 0, whiteSpace: "pre-line", fontSize: "var(--text-sm)" }}>{node.description}</p> : null}
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
                        const other = edge.source === node.id ? edge.target : edge.source;
                        return (
                          <li key={edge.id}>
                            <button onClick={() => pick(other)} title={edge.description ?? undefined}>
                              <span>{byId.get(other)?.name ?? other}</span>
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
                  <table className="table table-compact">
                    <thead>
                      <tr>
                        <th />
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
                <div>
                  <h3 className="field-label" style={{ margin: "0 0 6px" }}>
                    {t("graph.stats.evidence")}
                  </h3>
                  <table className="table table-compact">
                    <thead>
                      <tr>
                        <th />
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
            ) : null}
          </div>
        </section>
      </div>
      <Sheet title={t("graph.nodeTable")} pad={false}>
        <details>
          <summary style={{ padding: "12px 20px" }}>{t("graph.nodeTableShow", { count: data.nodes.length })}</summary>
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
                {[...data.nodes]
                  .sort((a, b) => b.degree - a.degree)
                  .map((n) => (
                    <tr key={n.id}>
                      <td>
                        <button className="link-button" onClick={() => pick(n.id)}>
                          {n.name}
                        </button>
                      </td>
                      <td>{n.type}</td>
                      <td className="num">{n.degree}</td>
                      <td>{n.community_id ? communityName(n.community_id) : "—"}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </details>
      </Sheet>
    </>
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
