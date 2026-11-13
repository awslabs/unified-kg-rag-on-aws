# Console design brief

The design the console's styles implement
(`console/web/src/styles/tokens.css`, `console/web/src/styles/app.css`).
Change this brief first when the direction changes.

## Subject and audience

A local workbench for one knowledge-graph RAG index: ask it questions and
audit every answer back to its passages and graph entities, build or update
the index, compare retrieval strategies with confidence intervals, inspect the
graph, watch cost and latency, and edit the config. The people at it are ML
and data engineers and solution architects deciding whether to trust a
strategy on their corpus, in English, Korean, Japanese, Chinese and six
European languages.

Each page has one job and one primary action:

| Page | Job | Primary action |
| --- | --- | --- |
| Ask | Get an answer and audit its evidence | Ask |
| Ingest | Build or update the index from a folder | Start ingestion |
| Evaluate | Compare strategies on a dataset | Start evaluation |
| Graph | Find an entity and read its neighbourhood | Find an entity |
| Dashboard | See what questions and jobs cost | (read only) |
| Settings | Change the config safely | Review changes |

## Direction: a developer tool

The console should feel like the tools its users already live in: a dark,
GitHub-like developer-tool palette (with a matching light theme), a sticky
top bar, cards on a quiet ground, compact tables and figures, a chat-style
answer view, and a force-directed canvas graph. The references are patterns,
not assets:

- **A dark, GitHub-like developer-tool palette**: near-black ground, card and
  raised layers a step lighter, hairline borders, one saturated accent,
  GitHub-style status colours with a label.
- **A top-bar shell**: the brand at the left, the pages as icon-and-label
  links with the current one filled in the accent, the index and the
  preferences at the right (the namespace chip's dot filled once the
  namespace holds an index, hollow while nothing is ingested, its name
  saying so too); when the row does not fit (measured at its content's own
  width: the labels' width follows the language), it drops the tagline, then the
  pickers' field labels, and only then folds the pages under a menu button.
- **A chat-style answer view**: the question as the asker's bubble at the
  right, the answer as a card at the left with its cited sources as chips,
  the composer docked at the foot of the conversation, a welcome with sample
  questions before the first one. The evidence trace stays beside it.
- **A force-directed canvas graph in the style of react-force-graph**: nodes
  sized by their connections (never under 2.5 px), coloured by community
  or type and edged in `--node-edge`, laid out warm from the server's packed
  layout with a weak pull toward the centre (so its islands stay beside the
  main component), labels set at 12 px in screen space, the focused,
  selected and traced nodes' always, the other lit nodes' most connected
  first at the first clear side of the node (right, left, above, below:
  inside the canvas, over no other label or drawn node) until
  labels cover 10% of the canvas (so zooming in labels more), the rest
  dimmed, directed edges with arrows that fade as they multiply, the edge's
  relationship shown on hover, a click that centres and zooms, a legend whose
  hovered entry lights its members and zoom tools over the canvas (the fit
  keeps the graph clear of them and of the search bar), and an inspector beside it.

## Tokens

Two first-class themes, each a full set of colour roles in
`console/web/src/styles/tokens.css` (light on `:root`, dark on
`:root[data-theme="dark"]`): the dark one is its own set, not an inversion.
The choice is System (the default: `prefers-color-scheme`), Light or Dark,
from the toggle in the top bar, kept in `localStorage`;
`console/web/public/theme.js` applies it before the first paint (a file,
since the page's CSP allows no inline script).

| Role | Light | Dark | Use |
| --- | --- | --- | --- |
| `--paper` | `#f6f7f9` | `#0e1117` | Page ground |
| `--sheet` | `#ffffff` | `#151a21` | Cards, panels, the top bar (dark) |
| `--sheet-sunk` | `#f1f3f6` | `#1c222a` | Wells: code, the composer, table heads |
| `--rule` | `#d9dde3` | `#2e353f` | Borders |
| `--ink` | `#1c2128` (16.2:1 on sheet) | `#e6e9ee` (14.4:1) | Text |
| `--graphite` | `#59636e` (6.1:1 sheet, 5.7:1 paper) | `#8f98a3` (6.0:1, 6.5:1) | Secondary text |
| `--control-edge` | `#848c96` (3.4:1) | `#6b7480` (3.6:1) | Field edges |
| `--primary-bg` / `-ink` | `#4f46e5` / white (6.3:1) | `#5b5bd6` / white (5.4:1) | The primary action, the current page |
| `--cobalt` | `#4f46e5` (6.3:1) | `#a5a8ff` (8.0:1) | Links, the trace, focus ring |
| `--trace-wash` / `--trace-ink` | `#eef0ff` / `#3730a3` (8.8:1) | `#232557` / `#c9ccff` (9.2:1) | Chosen chips, citation chips |
| `--highlighter` / `--mark-ink` | `#ffd33d` / `#1c2128` (11.3:1) | same | Matched evidence, cited markers |
| `--good` / `--warn` / `--bad` | 4.6 / 4.5 / 4.7:1 on their washes | 6.3 / 6.2 / 4.9:1 | Status, with a label |
| `--series-1..8` | dataviz reference, light steps | the same hues, dark steps (3:1 on sheet; 5 and 7 lighter, apart under colour-blindness) | Graph communities and types, charts |
| `--node-edge` | `--control-edge` (3.3:1 on the canvas) | `--canvas-bg` | A graph node's edge: carries the light theme's 3:1, where series 3-5 and other fall under it |

Type: Pretendard (SIL OFL 1.1), one design for Latin and Hangul, with
Pretendard JP for a Japanese page and the system's faces after them (Chinese
in PingFang SC or Microsoft YaHei); JetBrains Mono (OFL) for ids, strategies
and figures. Pretendard is served as its vendor ships it, unmodified (its
Reserved Font Name stays valid): the variable font's own dynamic-subset
stylesheet, where one file per unicode slice carries every weight, so a page
using 400 to 700 fetches each slice once (three static weights would fetch it
three times and add three stylesheets). Pretendard JP's stylesheet loads only
when the page turns Japanese. JetBrains Mono is subset from its @fontsource
package by `console/web/scripts/gen-fonts.mjs`. Licences are in
`console/web/public/licenses/`. Titles use the sans at 700 with -0.02em
tracking (Korean too; no serif display face, which clashed with Hangul).
Scale (px): 12 labels, 13 small, 14 body compact (tables, forms, nav), 15
answers, 18 card headings, 22 sections, 28 page titles; CJK text takes a 1.7
line height and Korean keeps words whole (`word-break: keep-all`). User
content (a question, an answer, a passage, a name) whose dominant script is
not the page's carries its own `lang`, so it breaks by its language's rules;
dominant, not any: a screen reader picks its voice by `lang`, and an English
answer quoting one Korean name must not be read in a Korean voice.

Space: a 4px base. Radius: 6px controls, 8px cards, pills for chips and
badges. Cards carry a hairline border and the faintest shadow; raised
elements (the composer, popovers, tooltips) a stronger one.

## Signature

The provenance trace: matched words in highlighter yellow, and an accent
rail whose stations are the sources an answer read, with arcs to the
passages that state the opened one, continued into the graph, where the
traced node wears a dashed ring and its edge carries moving particles.

## Avoid

- Copying any reference's code, strings, labels, data, logo or brand colours.
- A dark theme made by inverting or filtering the light one.
- More than one accent; colour as the only carrier of a status or a series.
- Decorative numbering, emoji icons, gradients, glow.
- Motion that is not tied to an event: an answer arriving (the rail draws,
  the sources settle in order) and the traced edge's particles; both are off
  under `prefers-reduced-motion`, where the graph opens already laid out.
