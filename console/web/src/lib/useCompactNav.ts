// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { useEffect, useLayoutEffect, useRef, useState, type DependencyList, type MutableRefObject, type RefObject } from "react";

/**
 * How far the top bar's row (brand, page links, tools) must compact to fit
 * ``row``, in steps: 0 shows everything, 1 hides the brand's tagline, 2 also
 * the pickers' field labels, and 3 (``NAV_FOLDED``) folds the links and tools
 * into the menu. Each step is measured, not guessed from the viewport: the
 * links' width follows the language (German labels are a third longer than
 * English ones) and the namespace chip. ``deps`` are what changes those
 * widths; the row is measured laid out at each step (``data-fold`` on its
 * header) before paint whenever they change or the web fonts finish loading,
 * and the widths are compared with the row's on resize.
 */
export const NAV_FOLDED = 3;

export function useCompactNav(row: RefObject<HTMLElement | null>, deps: DependencyList): number {
  const [fold, setFold] = useState(0);
  const needed = useRef<number[]>([]);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useLayoutEffect(() => measure(row.current, needed, setFold), deps);

  // The web fonts arrive after the first paint and widen the labels.
  useEffect(() => {
    const fonts = typeof document === "undefined" ? undefined : document.fonts;
    if (!fonts) return;
    let live = true;
    const again = () => {
      if (live) measure(row.current, needed, setFold);
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
    const observer = new ResizeObserver(() => setFold(fitting(needed.current, element.clientWidth)));
    observer.observe(element);
    return () => observer.disconnect();
  }, [row]);

  return fold;
}

/** The first step whose measured width fits ``width``; folded when none does. */
export function fitting(needed: readonly number[], width: number): number {
  const step = needed.findIndex((each) => each <= width);
  return step < 0 ? NAV_FOLDED : step;
}

/** Lay ``element``'s row out at every step short of folding (``data-fold`` on
 * its header), keep the widths each needs in ``needed``, and pick the first
 * that fits. */
function measure(element: HTMLElement | null, needed: MutableRefObject<number[]>, setFold: (fold: number) => void): void {
  const header = element?.closest<HTMLElement>("[data-fold]");
  if (!element || !header) return;
  const shown = header.dataset.fold;
  needed.current = Array.from({ length: NAV_FOLDED }, (_, step) => {
    header.dataset.fold = String(step);
    return element.scrollWidth;
  });
  header.dataset.fold = shown;
  setFold(fitting(needed.current, element.clientWidth));
}
