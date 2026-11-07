// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

// `npm run dev` serves the client on 5173 and forwards /api to a running
// run-console (CONSOLE_API, default http://127.0.0.1:8501).
const api = process.env.CONSOLE_API ?? "http://127.0.0.1:8501";

export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
    // Fonts stay files (never inlined): the CSP allows font-src 'self'.
    assetsInlineLimit: 0,
    // One bundle with every locale's catalog (about 0.6 MB, 0.2 MB gzipped).
    chunkSizeWarningLimit: 900,
  },
  server: {
    host: "127.0.0.1",
    proxy: { "/api": { target: api, changeOrigin: false } },
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    css: false,
    globals: true,
  },
});
