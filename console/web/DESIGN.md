# Console design brief

The design the console's styles implement
(`console/web/src/styles/tokens.css`, `console/web/src/styles/app.css`). Change this brief first when the direction changes.

## Subject and audience

A local workbench for one knowledge-graph RAG index: ask it questions and
audit every answer back to its passages and graph entities, build or update
the index, compare retrieval strategies with confidence intervals, inspect the
graph, watch cost and latency, and edit the config. The people at it are ML
and data engineers and solution architects deciding whether to trust a
strategy on their corpus. They read numbers and identifiers all day, in
English, Korean, Japanese, Chinese and six European languages.

Each page has one job and one primary action:

| Page | Job | Primary action |
| --- | --- | --- |
| Ask | Get an answer and audit its evidence | Ask |
| Ingest | Build or update the index from a folder | Start ingestion |
| Evaluate | Compare strategies on a dataset | Start evaluation |
| Graph | Find an entity and read its neighbourhood | Find an entity |
| Dashboard | See what questions and jobs cost | (read only) |
| Settings | Change the config safely | Review changes |

## Aesthetic family: the evidence ledger

An assay report or a survey field book: ruled, exact, quiet, with one ink
colour that owns the page and a highlighter that marks what matters now.
It is a working instrument, so density is a feature and decoration is not.

References (patterns, not assets):

- **IBM Carbon** (carbondesignsystem.com): the productive type set on IBM Plex
  (fixed heading sizes for dense product pages, `label-01` / `code-01` at
  12/16), layers instead of drop shadows, data-table row heights (32/40/48).
- **GitHub Primer** (primer.style): the Blankslate empty state (what this
  is, how to start, one action), underline tabs, progressive disclosure with
  native `details`.
- **Langfuse and Arize Phoenix** (LLM tracing and experiments): a trace read
  as a vertical list of spans beside its output, and runs compared against a
  named baseline column.
- **Neo4j Bloom and sigma.js** (graph explorers): the scene comes first; the
  search bar and the legend float over the canvas, an inspector card sits
  beside it.

## Tokens

Two first-class themes, each a full set of colour roles in
`console/web/src/styles/tokens.css` (light on `:root`, dark on
`:root[data-theme="dark"]`): the dark one is the same roles stepped for a
blue-black ground, not an inversion. The choice is System (the default:
`prefers-color-scheme`), Light or Dark, from the toggle in the rail, kept
in `localStorage`; `console/web/public/theme.js` applies it before the
first paint (a file, since the page's CSP allows no inline script).

Colour: dominant ink with a sharp accent; status colours stay functional.
Ratios are WCAG contrast against the surface named.

| Role | Light | Dark | Use |
| --- | --- | --- | --- |
| `--rail-bg` | `#0e2341` prussian | `#070d16` | The navigation rail, the dominant ink |
| `--rail-ink` | `#c9d3e0` (10.4:1) | `#c3cfdd` (12.3:1) | Text on the rail |
| `--primary-bg` / `-ink` | `#0e2341` / `#ffffff` (15.7:1) | `#b9cdf7` / `#0e2341` (9.8:1) | The primary action, selected tags |
| `--cobalt` | `#2453c7` (6.7:1 on sheet) | `#84a6f2` (7.1:1) | The trace, links, focus ring |
| `--highlighter` / `--mark-ink` | `#ffd43b` / `#111a24` (12.3:1) | same | Matched evidence, the current page, cited markers |
| `--paper` | `#f1f3f0` | `#0c131c` | Page ground (a ledger grey-green; blue-black) |
| `--sheet` | `#ffffff` | `#131c27` | Working surfaces |
| `--ink` | `#111a24` (15.7:1 on paper) | `#e4e9ef` (15.3:1 on paper, 14.1:1 on sheet) | Text |
| `--graphite` | `#56606b` (5.7:1 paper, 6.4:1 sheet) | `#9ba7b5` (7.6:1, 7.0:1) | Secondary text |
| `--rule` | `#d5dad6` | `#243142` | Hairline rules |
| `--control-edge` | `#7c8681` (3.4:1 paper, 3.8:1 sheet) | `#6c7d91` (4.4:1, 4.1:1) | Field edges |
| `--good` / `--warn` / `--bad` | 5.1 / 5.6 / 5.6:1 on their washes | 8.2 / 8.5 / 7.3:1 | Status, with a label |
| `--series-1..8` | dataviz reference, light steps | the same hues, dark steps (all 3:1 on sheet) | Graph communities and types |

Type: one superfamily, IBM Plex (SIL OFL 1.1), self-hosted as WOFF2 subsets
(`console/web/scripts/gen-fonts.mjs` writes the generated `fonts.css` next
to the tokens; the licences are in `console/web/public/licenses/`).

| Role | Face | Weights | Use |
| --- | --- | --- | --- |
| Display | IBM Plex Serif | 500 | Page titles only (Latin scripts) |
| Body | IBM Plex Sans (+ Cyrillic) | 400, 600 | Everything else |
| Data | IBM Plex Mono | 400, 600 | Identifiers, strategies, metrics, figures in tables |
| Korean | IBM Plex Sans KR | 400, 600 | Hangul, same design as Plex Sans |
| Japanese | IBM Plex Sans JP | 400, 600 | Kana and kanji |
| Chinese | System (PingFang SC, Microsoft YaHei, Noto Sans SC) | | No Plex SC is published as a web font package |

Scale (px, line height): 12/16 label and caption, 13/20 small, 14/20 body
compact (tables, forms), 16/24 body, 20/28 sheet heading, 28/36 section,
40/48 page title. CJK body text takes a 1.75 line height and Korean keeps
words whole (`word-break: keep-all`). Weights jump 400 to 600; nothing
in between.

Space: a 4px base (4, 8, 12, 16, 24, 32, 48). Radius: 2px on controls and
panels, round only for graph nodes and stations. No drop shadows on panels;
a raised element (a popover, a tooltip) gets one.

## Signature

The provenance trace: matched words in highlighter yellow, and a cobalt rail
whose stations are the sources an answer read, with arcs to the passages
that state the opened one. The same yellow marks where you are in the rail
navigation, so the accent always means "look here".

## Avoid

- Inter, Roboto, Arial, a system-font-only stack; the convergent display
  picks (Space Grotesk, Bricolage Grotesque).
- Purple or indigo gradients, glassmorphism, glow.
- Evenly spread pastel palettes; more than one accent.
- Every block a white rounded card with a drop shadow on grey; pill chips
  for everything.
- Decorative numbering (01 / 02 / 03), emoji, icon soup.
- The cream-and-serif editorial look, and near-black with an acid accent.
- A dark theme made by inverting or filtering the light one.
- Motion that is not tied to an event: the one orchestrated moment is an
  answer arriving (the rail draws, the sources settle in order), and it is
  off under `prefers-reduced-motion`.
