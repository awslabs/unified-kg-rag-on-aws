// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/** The namespace the header's picker chose, which Ask, Evaluate and Graph
 * read: kept in the browser across reloads, and dropped when the stores no
 * longer list it. */

const KEY = "console.namespace";

export function storedNamespace(): string | null {
  try {
    return window.localStorage.getItem(KEY);
  } catch {
    return null;
  }
}

export function storeNamespace(namespace: string | null): void {
  try {
    if (namespace) window.localStorage.setItem(KEY, namespace);
    else window.localStorage.removeItem(KEY);
  } catch {
    // A browser that keeps nothing picks the config's again on reload.
  }
}

/** The namespace the pages read: the one picked, while the picker offers it
 * (`offered`, the stores' and the config's; undefined until listed, when the
 * pick is kept, so a reload does not open the config's first), else the
 * config's (`fallback`). */
export function effectiveNamespace(picked: string | null, offered: string[] | undefined, fallback: string | null): string | null {
  if (!fallback) return null;
  if (picked && (offered === undefined || offered.includes(picked))) return picked;
  return fallback;
}
