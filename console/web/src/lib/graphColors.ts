// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
const SERIES = 8;

/** The series colour token of a legend slot; past eight, "other". */
export function colorSlot(index: number): string {
  return index < SERIES ? `var(--series-${index + 1})` : "var(--series-other)";
}
