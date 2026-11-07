// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/** The provenance trail the Graph page continues: the question and the
 * source the user followed from the Ask page. */
import { createContext, useContext, useState, type ReactNode } from "react";
import type { Source } from "./types";

export interface Trail {
  question: string;
  source: Source;
  sources: Source[];
}

const TrailContext = createContext<{ trail: Trail | null; setTrail: (t: Trail | null) => void } | null>(null);

export function TrailProvider({ children }: { children: ReactNode }) {
  const [trail, setTrail] = useState<Trail | null>(null);
  return <TrailContext.Provider value={{ trail, setTrail }}>{children}</TrailContext.Provider>;
}

export function useTrail() {
  const value = useContext(TrailContext);
  if (!value) throw new Error("useTrail outside TrailProvider");
  return value;
}
