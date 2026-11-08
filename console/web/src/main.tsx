// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// Self-hosted fonts (SIL OFL 1.1), bundled by the build: no request leaves
// the host. Pretendard is the vendor's own variable font and its own
// dynamic-subset stylesheet, unmodified (its Reserved Font Name stays
// valid); one file per slice carries every weight, and a browser fetches a
// slice only for the characters a page shows. Pretendard JP loads with the
// Japanese page only (lib/scriptFonts.ts). JetBrains Mono: fonts.css, which
// scripts/gen-fonts.mjs writes (npm run fonts).
import "pretendard/dist/web/variable/pretendardvariable-dynamic-subset.css";
import "./styles/fonts.css";
import "./styles/tokens.css";
import "./styles/app.css";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { I18nProvider } from "./i18n/i18n";
import { TrailProvider } from "./lib/trail";

const root = document.getElementById("root");
if (root) {
  createRoot(root).render(
    <StrictMode>
      <I18nProvider>
        <TrailProvider>
          <App />
        </TrailProvider>
      </I18nProvider>
    </StrictMode>,
  );
}
