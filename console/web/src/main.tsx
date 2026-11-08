// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// Self-hosted fonts (JetBrains Mono, OFL-1.1), bundled by the build: no request
// leaves the host. scripts/gen-fonts.mjs writes fonts.css (npm run fonts).
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
