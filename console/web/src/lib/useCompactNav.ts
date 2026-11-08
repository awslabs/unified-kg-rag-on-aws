// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { useEffect, useLayoutEffect, useRef, useState, type DependencyList, type MutableRefObject, type RefObject } from "react";

/**
 * Whether the top bar's row (brand, page links, tools) overflows ``row`` and
 * must collapse into the menu. Measured, not guessed from the viewport: the
 * links' width follows the language (German labels are a third longer than
 * English ones) and the namespace chip. ``deps`` are what changes that width;
 * the row is measured laid out in full (``data-compact`` off on its header)
 * before paint whenever they change or the web fonts finish loading, and
 * compared with its width on resize.
 */
export function useCompactNav(row: RefObject<HTMLElement | null>, deps: DependencyList): boolean {
  const [compact, setCompact] = useState(false);
  const needed = useRef(0);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useLayoutEffect(() => measure(row.current, needed, setCompact), deps);

  // The web fonts arrive after the first paint and widen the labels.
  useEffect(() => {
    const fonts = typeof document === "undefined" ? undefined : document.fonts;
    if (!fonts) return;
    let live = true;
    const again = () => {
      if (live) measure(row.current, needed, setCompact);
    };
    void fonts.ready.then(again);
    fonts.addEventListener("loadingdone", again);
    return () => {
      live = false;
      fonts.removeEventListener("loadingdone", again);
    };
  }, [row]);

  useEffect(() => {
    const element = row.current;
    if (!element || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => setCompact(needed.current > element.clientWidth));
    observer.observe(element);
    return () => observer.disconnect();
  }, [row]);

  return compact;
}

/** Lay ``element``'s row out in full (``data-compact`` off on its header),
 * keep the width it needs in ``needed``, and collapse when it does not fit. */
function measure(
  element: HTMLElement | null,
  needed: MutableRefObject<number>,
  setCompact: (compact: boolean) => void,
): void {
  const header = element?.closest<HTMLElement>("[data-compact]");
  if (!element || !header) return;
  const shown = header.dataset.compact;
  header.dataset.compact = "false";
  needed.current = element.scrollWidth;
  header.dataset.compact = shown;
  setCompact(needed.current > element.clientWidth);
}
