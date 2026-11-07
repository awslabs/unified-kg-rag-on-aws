// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/** A small History-API router: the server answers every page path with
 * index.html, so /ask, /graph?focus=... are real URLs. */
import { useEffect, useState } from "react";

export const PAGES = ["ask", "ingest", "evaluate", "graph", "dashboard", "settings"] as const;
export type Page = (typeof PAGES)[number];

export interface Route {
  page: Page;
  params: URLSearchParams;
}

export function parseRoute(location: Pick<Location, "pathname" | "search">): Route {
  const name = location.pathname.replace(/^\/+/, "").split("/")[0] ?? "";
  const page = (PAGES as readonly string[]).includes(name) ? (name as Page) : "ask";
  return { page, params: new URLSearchParams(location.search) };
}

const EVENT = "ukr-navigate";

export function navigate(page: Page, params?: Record<string, string>): void {
  const search = params && Object.keys(params).length ? `?${new URLSearchParams(params)}` : "";
  window.history.pushState(null, "", `/${page}${search}`);
  window.dispatchEvent(new Event(EVENT));
}

export function useRoute(): Route {
  const [route, setRoute] = useState(() => parseRoute(window.location));
  useEffect(() => {
    const update = () => setRoute(parseRoute(window.location));
    window.addEventListener("popstate", update);
    window.addEventListener(EVENT, update);
    return () => {
      window.removeEventListener("popstate", update);
      window.removeEventListener(EVENT, update);
    };
  }, []);
  return route;
}
