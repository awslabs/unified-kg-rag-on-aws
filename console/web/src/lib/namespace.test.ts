// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { describe, expect, it } from "vitest";
import { effectiveNamespace } from "./namespace";

describe("the namespace the pages read", () => {
  it("is the one picked while the picker offers it, else the config's", () => {
    expect(effectiveNamespace("v2", ["default", "v2"], "default")).toBe("v2");
    // Gone from the stores (or not listed yet): the config's.
    expect(effectiveNamespace("v2", ["default"], "default")).toBe("default");
    // Not listed yet: the pick is kept (a reload opens it, not the config's first).
    expect(effectiveNamespace("v2", undefined, "default")).toBe("v2");
    // No config loaded: none.
    expect(effectiveNamespace("v2", ["v2"], null)).toBeNull();
    expect(effectiveNamespace(null, ["default", "v2"], "default")).toBe("default");
  });
});
