// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
import { useI18n } from "../i18n/i18n";
import type { StrategyInfo } from "../lib/types";

const GROUPS = ["graphrag", "lightrag", "hipporag", "raptor"] as const;

/** Strategies grouped by methodology; one the namespace cannot serve is
 * disabled, with the reason. `multiple` picks several (an evaluation). */
export function StrategyPicker({
  strategies,
  value,
  onChange,
  multiple = false,
  includeAuto = true,
}: {
  strategies: StrategyInfo[];
  value: string[];
  onChange: (value: string[]) => void;
  multiple?: boolean;
  includeAuto?: boolean;
}) {
  const { t } = useI18n();
  const toggle = (name: string) => {
    if (!multiple) return onChange([name]);
    onChange(value.includes(name) ? value.filter((v) => v !== name) : [...value, name]);
  };
  const auto = strategies.find((s) => s.name === "auto");
  const unavailable = strategies.filter((s) => !s.available && s.name !== "auto");
  return (
    <div className="stack" style={{ gap: 10 }}>
      <div role="group" aria-label={t("strategy.label")} className="stack" style={{ gap: 8 }}>
        {includeAuto && auto ? (
          <div className="chips">
            <Chip info={auto} selected={value.includes("auto")} multiple={multiple} onClick={() => toggle("auto")} />
            <span className="field-help" style={{ alignSelf: "center" }}>
              {t("strategy.autoHelp", { list: strategies.filter((s) => s.routable).map((s) => s.name).join(", ") })}
            </span>
          </div>
        ) : null}
        {GROUPS.map((group) => {
          const members = strategies.filter((s) => s.methodology === group);
          if (!members.length) return null;
          return (
            <div key={group} className="row" style={{ alignItems: "baseline" }}>
              <span className="field-help" style={{ minWidth: "6.5rem" }}>
                {t(`methodology.${group}`)}
              </span>
              <div className="chips">
                {members.map((info) => (
                  <Chip key={info.name} info={info} selected={value.includes(info.name)} multiple={multiple} onClick={() => toggle(info.name)} />
                ))}
              </div>
            </div>
          );
        })}
      </div>
      {value.length === 1 && value[0] ? <p className="field-help" style={{ margin: 0 }}>{t(`strategy.help.${value[0]}`)}</p> : null}
      {unavailable.length ? (
        <details>
          <summary className="field-help">{t("strategy.unavailable", { count: unavailable.length })}</summary>
          <ul className="field-help">
            {unavailable.map((s) => (
              <li key={s.name}>
                <span className="mono">{s.name}</span>: {s.reason}
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </div>
  );
}

function Chip({ info, selected, onClick }: { info: StrategyInfo; selected: boolean; multiple: boolean; onClick: () => void }) {
  const { t } = useI18n();
  return (
    <button
      type="button"
      className="chip mono"
      aria-pressed={selected}
      disabled={!info.available}
      title={info.available ? t(`strategy.help.${info.name}`) : (info.reason ?? undefined)}
      onClick={onClick}
    >
      {info.name}
    </button>
  );
}
