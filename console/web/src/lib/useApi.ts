// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { useCallback, useEffect, useRef, useState } from "react";

/** Load `load()` on mount and when `deps` change; `reload` loads again.
 * With `intervalMs`, polls while `poll` is true. */
export function useApi<T>(
  load: () => Promise<T>,
  deps: unknown[],
  options: { intervalMs?: number; poll?: boolean } = {},
) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const loader = useRef(load);
  useEffect(() => {
    loader.current = load;
  });
  const reload = useCallback(async () => {
    try {
      const value = await loader.current();
      setData(value);
      setError(null);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    setLoading(true);
    void reload();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  const { intervalMs, poll } = options;
  useEffect(() => {
    if (!intervalMs || !poll) return;
    const id = window.setInterval(() => void reload(), intervalMs);
    return () => window.clearInterval(id);
  }, [intervalMs, poll, reload]);
  return { data, error, loading, reload, setData };
}
