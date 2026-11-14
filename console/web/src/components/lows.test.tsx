// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { modelIds } from "../App";
import { I18nProvider, makeI18n, translate } from "../i18n/i18n";
import { contentLang } from "../lib/contentLang";
import type { Source } from "../lib/types";
import { EvidenceTrace } from "./EvidenceTrace";
import { isBudgetStop, stageName } from "./JobPanel";
import { formatBytes } from "./ui";
import { fitting, NAV_FOLDED } from "../lib/useCompactNav";
import { BudgetNotice, questionTitle, runModelCalls } from "../pages/IngestPage";
import type { JobRecord } from "../lib/types";

describe("small copy", () => {
  it("names every stage in every locale, an unknown one by its id", () => {
    expect(stageName((k) => translate("ko", k), "community_detection")).toBe("커뮤니티 탐지");
    expect(stageName((k) => translate("en", k), "document_parsing")).toBe("Document parsing");
    expect(stageName((k) => translate("en", k), "new_stage")).toBe("new_stage");
  });

  it("lists the distinct model ids, not the provider", () => {
    expect(modelIds({ answer_generation: "qwen2.5:7b", entity_extraction: "qwen2.5:7b", embedding: "bge-m3" })).toBe("qwen2.5:7b · bge-m3");
    expect(modelIds(null)).toBe("");
  });

  it("titles the integrity question by its reason, in every locale", () => {
    const titles = (["settings", "corpus", "cache", null] as const).map((reason) => questionTitle({ confirmation: "integrity", confirmation_reason: reason }));
    expect(titles).toEqual(["ingest.question.integritySettings", "ingest.question.integrityCorpus", "ingest.question.integrityCache", "ingest.question.integrity"]);
    expect(questionTitle({ confirmation: "reset", confirmation_reason: null })).toBe("ingest.question.reset");
    for (const locale of ["en", "ko", "ja", "zh", "de"] as const) {
      const shown = titles.map((key) => translate(locale, key));
      expect(new Set(shown).size, locale).toBe(titles.length);
      for (const [i, text] of shown.entries()) expect(text, locale).not.toBe(titles[i]);
    }
    expect(translate("en", titles[0] ?? "")).toBe("Loading settings changed since its last run");
  });

  it("offers to resume a run stopped at its budget, in the page's language", () => {
    // A run stopped at processing.max_run_* exits 1 (failed) with the
    // limit_exceeded class: its finished stages are kept, so running it
    // again resumes it.
    const job = { kind: "ingest", status: "failed", result: { error_class: "limit_exceeded" } } as unknown as JobRecord;
    let resumed = 0;
    const { rerender } = render(
      <I18nProvider initial="ko">
        <BudgetNotice job={job} disabled={false} onResume={() => (resumed += 1)} />
      </I18nProvider>,
    );
    expect(screen.getByText(translate("ko", "ingest.budget.title"))).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: translate("ko", "ingest.budget.resume") }));
    expect(resumed).toBe(1);
    for (const locale of ["en", "ko", "ja", "zh", "de", "es", "fr", "it", "pt", "ru"] as const) {
      for (const key of ["ingest.budget.title", "ingest.budget.help", "ingest.budget.resume"]) {
        expect(translate(locale, key), `${locale} ${key}`).not.toBe(locale === "en" ? key : translate("en", key));
      }
    }
    // Another failure, or a run that ended otherwise, offers nothing.
    for (const other of [{ kind: "ingest", status: "failed", result: { error_class: "failure" } }, { kind: "ingest", status: "succeeded", result: { error_class: "limit_exceeded" } }]) {
      rerender(
        <I18nProvider initial="ko">
          <BudgetNotice job={other as unknown as JobRecord} disabled={false} onResume={() => undefined} />
        </I18nProvider>,
      );
      expect(screen.queryByText(translate("ko", "ingest.budget.title"))).not.toBeInTheDocument();
    }
  });

  it("says a budget stop once and counts its model calls from its stages", () => {
    // Regression (W1 review D10): the generic "The run failed" notice stood
    // above the budget notice, and the calls read "-" (a stopped run's
    // status has no token_usage).
    const stopped = {
      kind: "ingest",
      status: "failed",
      result: { error_class: "limit_exceeded", status: "failed" },
      progress: {
        done: 2,
        total: 5,
        stages: [
          { name: "graph_extraction", status: "stopped", llm_calls: 4 },
          { name: "text_chunking", status: "restored", llm_calls: 9 },
          { name: "document_parsing", status: "completed", llm_calls: null },
        ],
      },
    } as unknown as JobRecord;
    expect(isBudgetStop(stopped)).toBe(true);
    expect(isBudgetStop({ ...stopped, kind: "evaluate" } as JobRecord)).toBe(false);
    expect(isBudgetStop({ ...stopped, result: { error_class: "failure" } } as JobRecord)).toBe(false);
    // This run's calls: a restored stage's were an earlier run's.
    expect(runModelCalls(stopped)).toBe(4);
    const finished = { ...stopped, status: "succeeded", result: { token_usage: { llm_calls: 12 } } } as unknown as JobRecord;
    expect(runModelCalls(finished)).toBe(12);
  });

  it("puts a byte count in the plural", () => {
    const { number } = makeI18n("en", () => undefined);
    expect(formatBytes(180, number)).toBe("180 bytes");
    expect(formatBytes(2048, number)).toBe("2 kB");
  });
});

describe("EvidenceTrace details", () => {
  it("folds a source's ids and fused score under a toggle", () => {
    const source: Source = {
      rank: 1, kind: "entity", id: "0d4c-uuid", title: "Vendor Corp", score: 0.0164, excerpt: "It ships parts.", truncated: false, document_ids: [],
      file_name: null, chunk_id: null, text_unit_ids: ["t1", "t2"], source_kinds: [], iri: null, provenance: null, marker: null,
    };
    render(
      <I18nProvider initial="en">
        <EvidenceTrace sources={[source]} question="Vendor" />
      </I18nProvider>,
    );
    fireEvent.click(screen.getByText("Vendor Corp").closest("button") as HTMLElement);
    const details = screen.getByText("Ids and score").closest("details") as HTMLDetailsElement;
    expect(details.open).toBe(false);
    expect(details).toContainElement(screen.getByText("0d4c-uuid"));
    expect(details).toContainElement(screen.getByText("2 passages"));
  });
});

describe("contentLang", () => {
  it("marks content whose script is not the page's", () => {
    expect(contentLang("케스트럴 창고까지 운송", "en")).toBe("ko");
    expect(contentLang("ケストレル倉庫", "ko")).toBe("ja");
    expect(contentLang("北京仓库", "ko")).toBe("zh");
    // Han alone on a Japanese page may be a Japanese name.
    expect(contentLang("東京倉庫", "ja")).toBeUndefined();
    expect(contentLang("케스트럴 창고", "ko")).toBeUndefined();
    expect(contentLang("Kestrel Depot", "ko")).toBeUndefined();
    expect(contentLang(null, "en")).toBeUndefined();
  });

  it("follows the dominant script, not one quoted name", () => {
    expect(contentLang("Umbra Freight carries the lidar units to Kestrel Depot (케스트럴 창고) within 10 business days.", "en")).toBeUndefined();
    expect(contentLang("The supplier is Globex Sensors (グローベックス・センサーズ), which warrants each unit.", "en")).toBeUndefined();
    expect(contentLang("Umbra Freight가 라이다 장치를 케스트럴 창고까지 운송합니다.", "en")).toBe("ko");
  });

  it("weighs words, not letters: a CJK sentence quoting English names is marked", () => {
    // Counted letter for letter, the quoted names outnumber the sentence.
    expect(contentLang("Umbra Freight는 10영업일 이내에 Kestrel Depot로 lidar unit을 배송합니다.", "en")).toBe("ko");
    expect(contentLang("Globex SensorsはOrion Labsにlidar unitを供給します。", "en")).toBe("ja");
    // A name the particle is written onto takes part in the CJK sentence.
    expect(contentLang("Acme Robotics와 Orion Labs가 맺은 supply agreement의 penalty clause는 무엇인가요?", "ja")).toBe("ko");
    expect(contentLang("Acme Robotics와 Orion Labs가 맺은 supply agreement의 penalty clause는 무엇인가요?", "ko")).toBeUndefined();
    // A tie goes to the CJK script, as in the backend.
    expect(contentLang("Kestrel 창고", "en")).toBe("ko");
    // Spaced English quoting a CJK name stays the page's.
    expect(contentLang("Ask Kestrel Depot (케스트럴 창고) about the 物流 schedule of Umbra Freight.", "en")).toBeUndefined();
  });
});

describe("the top bar's steps", () => {
  it("takes the first step that fits, then folds", () => {
    // Widths the row needs in full, without the tagline, without the labels.
    const needed = [1520, 1300, 1230];
    expect(fitting(needed, 1600)).toBe(0);
    expect(fitting(needed, 1440)).toBe(1);
    expect(fitting(needed, 1250)).toBe(2);
    expect(fitting(needed, 1000)).toBe(NAV_FOLDED);
    expect(fitting([], 1000)).toBe(NAV_FOLDED);
  });
});
