// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
// Sets <html data-theme> before the first paint, so the page never flashes
// the other theme: the saved choice (src/lib/theme.ts, the same key), else
// the operating system's. A file, not an inline script: the console's CSP
// allows scripts from its own origin only.
(function () {
  var choice = null;
  try {
    choice = window.localStorage.getItem("ukr-console-theme");
  } catch (error) {
    choice = null;
  }
  var dark =
    choice === "dark" ||
    (choice !== "light" && !!window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
})();
