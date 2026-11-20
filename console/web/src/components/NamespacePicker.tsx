// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { useI18n } from "../i18n/i18n";

/** The header's namespace picker: the namespace Ask, Evaluate and Graph
 * read, of those the stores hold (the config's first). Its dot says whether
 * the chosen one holds an index. */
export function NamespacePicker({
  value,
  namespaces,
  indexed,
  onChange,
}: {
  value: string;
  namespaces: string[];
  indexed: boolean | null;
  onChange: (namespace: string) => void;
}) {
  const { t } = useI18n();
  const id = "namespace-picker";
  const state = indexed === false ? t("shell.namespaceEmpty") : t("shell.namespacePick");
  const options = namespaces.includes(value) ? namespaces : [value, ...namespaces];
  return (
    <span className="namespace-picker" title={state}>
      <label htmlFor={id}>
        {/* Named, not labelled on screen: the chip stays as narrow as the
            namespace it shows, so the top bar keeps its links inline. */}
        <span className="visually-hidden">{t("shell.namespace")}</span>
        <span className="namespace-chip" data-indexed={indexed === null ? undefined : String(indexed)}>
          <select id={id} className="namespace-select" value={value} aria-describedby={`${id}-state`} onChange={(event) => onChange(event.target.value)}>
            {options.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        </span>
      </label>
      <span id={`${id}-state`} className="visually-hidden">
        {state}
      </span>
    </span>
  );
}
