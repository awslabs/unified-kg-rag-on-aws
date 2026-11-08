// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { useI18n } from "../i18n/i18n";
import { THEME_CHOICES, useTheme } from "../lib/theme";

/** System / Light / Dark, as one group of toggle buttons. */
export function ThemePicker() {
  const { t } = useI18n();
  const { choice, setChoice } = useTheme();
  return (
    <div className="field">
      <span className="field-label" id="theme-label">
        {t("shell.theme")}
      </span>
      <div className="segmented" role="group" aria-labelledby="theme-label">
        {THEME_CHOICES.map((option) => (
          <button key={option} type="button" aria-pressed={choice === option} onClick={() => setChoice(option)}>
            {t(`shell.theme.${option}`)}
          </button>
        ))}
      </div>
    </div>
  );
}
