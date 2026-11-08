// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * An answer with its citation markers: each [n] the answer's citations hold
 * opens its source in the provenance trace; a number the context did not
 * show is marked as such. While the answer streams, its markers are shown
 * but not yet linked (the citations come with the result). An answer whose
 * strategy does not cite (`search.citations`) is plain text.
 */
import { useI18n } from "../i18n/i18n";
import { answerSegments } from "../lib/citations";
import type { Citation } from "../lib/types";

export interface AnswerTextProps {
  text: string;
  /** The answer's citations; undefined while it streams. */
  citations?: Citation[];
  onOpen?: (citation: Citation) => void;
  /** Whether the answer's strategy cites; false shows its text as written. */
  cites?: boolean;
}

export function AnswerText({ text, citations, onOpen, cites = true }: AnswerTextProps) {
  const { t } = useI18n();
  const byMarker = new Map((citations ?? []).map((c) => [c.marker, c]));
  if (!cites) return <span>{text}</span>;
  return (
    <>
      {answerSegments(text).map((segment, index) => {
        if ("text" in segment) return <span key={index}>{segment.text}</span>;
        if (!citations) {
          return (
            <span key={index} className="cite cite-pending">
              {segment.marker}
            </span>
          );
        }
        return (
          <span key={index} className="cite-group">
            {segment.numbers.map((n, i) => {
              const citation = byMarker.get(n);
              if (!citation) {
                return (
                  <span key={i} className="cite cite-unknown" title={t("ask.cite.unknown", { marker: n })}>
                    [{n}]
                  </span>
                );
              }
              if (citation.source_rank == null || !onOpen) {
                return (
                  <span key={i} className="cite" title={citationLabel(citation) ?? undefined}>
                    [{n}]
                  </span>
                );
              }
              return (
                <button
                  key={i}
                  type="button"
                  className="cite cite-link"
                  aria-label={t("ask.cite.open", { marker: n, source: citationLabel(citation) ?? `#${citation.source_rank}` })}
                  onClick={() => onOpen(citation)}
                >
                  [{n}]
                </button>
              );
            })}
          </span>
        );
      })}
    </>
  );
}

/** What a citation names: its document (file and pages) or its section. */
export function citationLabel(citation: Citation): string | null {
  return citation.file_path ?? citation.file_name ?? citation.title ?? citation.source_id;
}

/** The cited pages: "3" or "3–5", null when unknown. */
export function citationPages(citation: Citation): string | null {
  if (citation.page == null) return null;
  return citation.page_end != null && citation.page_end !== citation.page ? `${citation.page}–${citation.page_end}` : String(citation.page);
}

export function CitationList({ citations, onOpen }: { citations: Citation[]; onOpen?: (citation: Citation) => void }) {
  const { t } = useI18n();
  const cited = citations.filter((c) => c.cited);
  if (!cited.length) return null;
  return (
    <div className="citations">
      <span className="citations-title">{t("ask.citations.title")}</span>
      <ol className="citations-list">
        {cited.map((citation) => {
          const pages = citationPages(citation);
          const label = citationLabel(citation) ?? t("ask.trace.untitled");
          const body = (
            <>
              <span className="cite">[{citation.marker}]</span> <span className="citation-label">{label}</span>
              {citation.title && citation.title !== label ? <span className="muted"> · {citation.title}</span> : null}
              {pages ? <span className="muted"> · {t("ask.citations.pages", { pages })}</span> : null}
              {citation.source_rank == null ? <span className="muted"> · {t("ask.citations.summary")}</span> : null}
            </>
          );
          return (
            <li key={citation.marker}>
              {citation.source_rank != null && onOpen ? (
                <button type="button" className="link-button citation-link" onClick={() => onOpen(citation)}>
                  {body}
                </button>
              ) : (
                body
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
}
