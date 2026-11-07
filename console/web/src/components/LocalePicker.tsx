// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { isLocale, LOCALES, useI18n } from "../i18n/i18n";

export function LocalePicker({ id = "locale" }: { id?: string }) {
  const { t, locale, setLocale } = useI18n();
  return (
    <label className="field" htmlFor={id}>
      <span className="field-label">{t("shell.language")}</span>
      <select
        id={id}
        className="select"
        value={locale}
        onChange={(event) => {
          if (isLocale(event.target.value)) setLocale(event.target.value);
        }}
      >
        {Object.entries(LOCALES).map(([code, info]) => (
          <option key={code} value={code} lang={info.tag}>
            {info.name}
          </option>
        ))}
      </select>
    </label>
  );
}
