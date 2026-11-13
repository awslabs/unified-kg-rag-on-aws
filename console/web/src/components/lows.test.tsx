// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { modelIds } from "../App";
import { I18nProvider, makeI18n, translate } from "../i18n/i18n";
import { contentLang } from "../lib/contentLang";
import type { Source } from "../lib/types";
import { EvidenceTrace } from "./EvidenceTrace";
import { stageName } from "./JobPanel";
import { formatBytes } from "./ui";
import { fitting, NAV_FOLDED } from "../lib/useCompactNav";

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
