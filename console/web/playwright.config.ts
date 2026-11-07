// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { defineConfig } from "@playwright/test";

// The smoke test runs against a console that is already running
// (CONSOLE_URL; console/demo/demo.py --models fake starts one).
export default defineConfig({
  testDir: "e2e",
  timeout: 90_000,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: process.env.CONSOLE_URL ?? "http://127.0.0.1:8501",
    reducedMotion: "reduce",
    launchOptions: process.env.CHROMIUM_PATH ? { executablePath: process.env.CHROMIUM_PATH } : {},
  },
});
