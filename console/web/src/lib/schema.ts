// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/** Walking the JSON schema of the server's Config model. */

export type JsonSchema = {
  $ref?: string;
  $defs?: Record<string, JsonSchema>;
  type?: string | string[];
  properties?: Record<string, JsonSchema>;
  items?: JsonSchema;
  additionalProperties?: JsonSchema | boolean;
  anyOf?: JsonSchema[];
  allOf?: JsonSchema[];
  enum?: unknown[];
  const?: unknown;
  default?: unknown;
  description?: string;
  title?: string;
  format?: string;
  minimum?: number;
  maximum?: number;
  exclusiveMinimum?: number;
};

export type FieldKind = "boolean" | "integer" | "number" | "string" | "enum" | "json" | "secret" | "section";

export interface SchemaField {
  path: string;
  key: string;
  kind: FieldKind;
  description: string;
  defaultValue: unknown;
  hasDefault: boolean;
  options: unknown[];
  nullable: boolean;
  children: SchemaField[];
}

function resolve(root: JsonSchema, node: JsonSchema): JsonSchema {
  let current = node;
  for (let i = 0; i < 10 && current.$ref; i += 1) {
    const name = current.$ref.replace("#/$defs/", "");
    const target = root.$defs?.[name] ?? {};
    const rest: JsonSchema = { ...current };
    delete rest.$ref;
    current = { ...target, ...rest, description: rest.description ?? target.description };
  }
  if (current.allOf?.length === 1 && current.allOf[0]) {
    const inner = resolve(root, current.allOf[0]);
    current = { ...inner, ...current, allOf: undefined, description: current.description ?? inner.description };
  }
  return current;
}

function kindOf(root: JsonSchema, node: JsonSchema): { kind: FieldKind; nullable: boolean; options: unknown[]; inner: JsonSchema } {
  const resolved = resolve(root, node);
  let options = resolved.anyOf?.map((o) => resolve(root, o)) ?? [resolved];
  const nullable = options.some((o) => o.type === "null");
  options = options.filter((o) => o.type !== "null");
  const inner = options.length === 1 && options[0] ? { ...options[0], description: resolved.description ?? options[0].description, default: resolved.default } : resolved;
  if (inner.format === "password" || options.some((o) => o.format === "password")) return { kind: "secret", nullable, options: [], inner };
  if (inner.enum) return { kind: "enum", nullable, options: inner.enum, inner };
  if (options.length > 1) return { kind: "json", nullable, options: [], inner };
  if (inner.properties) return { kind: "section", nullable, options: [], inner };
  if (inner.type === "boolean") return { kind: "boolean", nullable, options: [], inner };
  if (inner.type === "integer") return { kind: "integer", nullable, options: [], inner };
  if (inner.type === "number") return { kind: "number", nullable, options: [], inner };
  if (inner.type === "string") return { kind: "string", nullable, options: [], inner };
  return { kind: "json", nullable, options: [], inner };
}

/** The config's fields as a tree (sections hold their fields). */
export function fieldsOf(root: JsonSchema, node: JsonSchema = root, prefix = "", depth = 0): SchemaField[] {
  const resolved = resolve(root, node);
  const fields: SchemaField[] = [];
  for (const [key, child] of Object.entries(resolved.properties ?? {})) {
    const path = prefix ? `${prefix}.${key}` : key;
    const { kind, nullable, options, inner } = kindOf(root, child);
    const childResolved = resolve(root, child);
    fields.push({
      path,
      key,
      kind,
      description: childResolved.description ?? inner.description ?? "",
      defaultValue: childResolved.default ?? inner.default,
      hasDefault: "default" in childResolved || "default" in inner,
      options,
      nullable,
      children: kind === "section" && depth < 8 ? fieldsOf(root, inner, path, depth + 1) : [],
    });
  }
  return fields;
}

export function flatten(fields: SchemaField[]): SchemaField[] {
  return fields.flatMap((f) => (f.kind === "section" ? flatten(f.children) : [f]));
}

export function valueAt(data: unknown, path: string): unknown {
  let current: unknown = data;
  for (const key of path.split(".")) {
    if (current && typeof current === "object" && key in (current as Record<string, unknown>)) {
      current = (current as Record<string, unknown>)[key];
    } else {
      return undefined;
    }
  }
  return current;
}

/** The value an editor's text stands for, or an error key. */
export function parseValue(field: SchemaField, text: string): { value?: unknown; error?: string } {
  if (field.nullable && text.trim() === "") return { value: null };
  switch (field.kind) {
    case "integer": {
      if (!/^-?\d+$/.test(text.trim())) return { error: "settings.errors.integer" };
      return { value: Number(text) };
    }
    case "number": {
      const value = Number(text);
      if (text.trim() === "" || Number.isNaN(value)) return { error: "settings.errors.number" };
      return { value };
    }
    case "json":
      try {
        return { value: JSON.parse(text) };
      } catch {
        return { error: "settings.errors.json" };
      }
    default:
      return { value: text };
  }
}
