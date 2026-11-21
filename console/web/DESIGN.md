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

## Direction: the calculation pad

The console should feel like an engineer's computation pad: a precise,
calm instrument for checking work, not a marketing site. Pad green and
pencil graphite, the highlighter as the only colour, a sticky top bar,
cards on the pad, compact tables and boxed figures, a chat-style answer
view, and a force-directed canvas graph on the pad's grid. The references
are patterns, not assets:

- **Pad and pencil**: a pale pad-green ground, cards a step lighter,
  hairline borders, ink and controls in pencil graphite (the primary action
  is a graphite button, links are graphite and underlined). The dark theme
  is its own set: carbon ground, chalk ink, the primary action in chalk.
- **The highlighter is the one colour**: it marks matched evidence, the
  current page and the chosen option, so yellow always says "this one".
  Status colours (with a label) and the graph's series are the only other
  hues.
- **Legible figures**: Atkinson Hyperlegible Next and Mono, drawn so that
  0/O, 1/l/I and 5/S never look alike, for the ids, hashes, token counts and
  costs the page is full of.
- **A top-bar shell**: the brand at the left, the pages as icon-and-label
  links with the current one highlighted, the index and the
  preferences at the right (the namespace chip's dot filled once the
  namespace holds an index, hollow while nothing is ingested, its name
  saying so too); when the row does not fit (measured at its content's own
  width: the labels' width follows the language), it drops the tagline, then the
  pickers' field labels, and only then folds the pages under a menu button.
- **A chat-style answer view**: the question as the asker's bubble at the
  right, the answer as a card at the left with its cited sources as chips,
  the composer docked at the foot of the conversation, a welcome with sample
  questions before the first one. The evidence trace stays beside it.
- **The pad's grid where something is worked out**: minor lines every 8 px
  and major ones every 40 px on the graph canvas, and a ruled composer, one
  line under each line of text; nowhere else.
- **A force-directed canvas graph in the style of react-force-graph**: nodes
  sized by their connections (never under 2.5 px), coloured by community
  or type and edged in `--node-edge`, laid out warm from the server's packed
  layout with a weak pull toward the centre (so its islands stay beside the
  main component), labels set at 12 px in screen space, the focused,
  selected and traced nodes' always, the other lit nodes' most connected
  first at the first clear side of the node (right, left, above, below:
  inside the canvas, over no other label or drawn node; when none is, over
  less connected nodes only, so a dense centre's hubs are named) until
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
| `--paper` | `#e8efe6` (pad green) | `#161718` (carbon) | Page ground |
| `--sheet` | `#fbfdfa` | `#1d1f21` | Cards, panels, the top bar |
| `--sheet-sunk` | `#eff4ec` | `#242729` | Wells: code, the composer, table heads, the question |
| `--rule` / `--rule-strong` | `#c6d3c2` / `#a6b7a1` | `#34383c` / `#464b50` | Borders; a figure's rule |
| `--ink` | `#1b211d` pencil graphite (16.0:1 on sheet, 14.0:1 on paper) | `#ecedea` chalk (14.1:1, 15.3:1) | Text, the double rule |
| `--graphite` | `#4f5b53` (7.0:1 sheet, 6.1:1 paper) | `#a3a8a4` (6.8:1, 7.4:1) | Secondary text, chart bars |
| `--control-edge` | `#77887a` (3.7:1, 3.2:1 on paper) | `#7a8087` (4.1:1) | Field edges |
| `--primary-bg` / `-ink` | `#1f2622` / white (15.5:1) | `#e6e8e4` / `#161718` (14.6:1) | The primary action |
| `--cobalt` (the accent) | `#1b211d` (16.0:1) | `#ffe07a` (12.8:1) | Links (underlined), the trace, focus ring |
| `--trace-wash` / `--trace-ink` | `#fff1a8` / `#1b211d` (14.4:1) | `#38321c` / `#ffe58a` (10.3:1) | Chosen chips and rows |
| `--highlighter` / `--mark-ink` | `#ffd43b` / `#1b211d` (11.5:1) | same | Matched evidence, the current page, cited markers, an opened source |
| `--pad-line` / `--pad-line-major` | graphite at 5% / 10% | chalk at 4% / 8% | The pad's grid and the composer's lines |
| `--result-rule` | `3px double var(--ink)` | the same, in chalk | The boxed result |
| `--good` / `--warn` / `--bad` | 4.6 / 5.2 / 5.5:1 on their washes | 6.9 / 7.3 / 6.0:1 | Status, with a label |
| `--series-1..8` | dataviz reference, light steps | the same hues, dark steps (3:1 on sheet; 5 and 7 lighter, apart under colour-blindness) | Graph communities and types, charts |
| `--node-edge` | `--control-edge` (3.6:1 on the canvas) | `--canvas-bg` | A graph node's edge: carries the light theme's 3:1, where series 3-5 and other fall under it |

Type: Atkinson Hyperlegible Next (SIL OFL 1.1, variable) for Latin text and
titles, Atkinson Hyperlegible Mono (OFL, variable) for ids, strategies and
figures; Pretendard (OFL) after them for Hangul, with Pretendard JP for a
Japanese page and the system's faces after them (Chinese in PingFang SC or
Microsoft YaHei), the CJK stacks as before. Atkinson has no Cyrillic: Russian
text takes the platform's sans, and the data face's Russian (chart units)
JetBrains Mono's Cyrillic subset (OFL). The Atkinson faces and JetBrains
Mono's Cyrillic are subset from their @fontsource packages by
`console/web/scripts/gen-fonts.mjs` (Latin and Latin Extended, upright
only). Pretendard is served as its vendor ships it, unmodified (its
Reserved Font Name stays valid): the variable font's own dynamic-subset
stylesheet, where one file per unicode slice carries every weight, so a page
using 400 to 700 fetches each slice once (three static weights would fetch it
three times and add three stylesheets). Pretendard JP's stylesheet loads only
when the page turns Japanese. Licences are in `console/web/public/licenses/`.
Titles use the sans at 800 with -0.025em tracking (Korean too; no serif
display face, which clashed with Hangul). Figures are set in the mono face,
tabular. Scale (px): 12 labels, 13 small, 14 body compact (tables, forms,
nav), 15 answers, 18 card headings, 22 sections and figures, 32 page titles; CJK text takes a 1.7
line height and Korean keeps words whole (`word-break: keep-all`). User
content (a question, an answer, a passage, a name) whose dominant script is
not the page's carries its own `lang`, so it breaks by its language's rules;
dominant, not any: a screen reader picks its voice by `lang`, and an English
answer quoting one Korean name must not be read in a Korean voice. Dominant
by words (`Intl.Segmenter`), not letters, as the backend's script counts
weigh letters by the tokens they carry: a Korean or Japanese sentence quoting
English names is the CJK language's, and a name its particle is written onto
("Labs가", "Sensorsは") counts with the CJK words; a tie goes to the CJK
script. Latin content on a CJK page is not marked: the script does not say
which language it is.

Space: a 4px base. Radius: 6px controls, 10px cards, pills for chips and
badges, 2px for a boxed figure. Cards carry a hairline border and no
shadow; raised elements (the composer, popovers, tooltips, the graph's
tools) a shadow. A dashboard figure is not a card: a rule above its label,
its value boxed.

## Signature

The boxed result: the double rule an engineer draws around a calculation's
final value, around the answer and around every dashboard figure, and
nothing else. Beside it, the provenance trace: matched words in the
highlighter, and a pencil rail whose stations are the sources an answer
read (an opened one takes the highlighter), with arcs to the passages that
state the opened one, continued into the graph, where the traced node wears
a dashed ring and its edge carries moving particles.

## Avoid

- Copying any reference's code, strings, labels, data, logo or brand colours.
- A dark theme made by inverting or filtering the light one.
- A hue accent beside the highlighter; colour as the only carrier of a
  status or a series; a link told from text by colour alone (links are
  underlined).
- The double rule on anything but the answer and a figure; the pad's grid
  beyond the graph and the composer.
- Decorative numbering, emoji icons, gradients, glow, paper textures.
- Motion that is not tied to an event: an answer arriving (the rail draws,
  the sources settle in order) and the traced edge's particles; both are off
  under `prefers-reduced-motion`, where the graph opens already laid out.
