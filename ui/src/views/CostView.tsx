/**
 * Running it here against paying for it.
 *
 * The inputs are assumptions, so they are editable and visible rather than
 * buried in a preferences pane: a comparison whose premises are hidden is a
 * claim, not a calculation. The one figure that is not an assumption — how
 * many hours of generation a month of this traffic takes — comes from the
 * measured speed, and carries its provenance like every other.
 */
import { useEffect, useState } from "react";
import * as engine from "../engine";
import type { CostComparison, CostQuery, RankedModel, Sizing } from "../engine";
import type { View } from "../App";
import * as fmt from "../format";
import { Card, Empty, Eyebrow, Figure, Spec } from "../components/ui";

const DEFAULTS: CostQuery = {
  requests: 3000,
  input: 8000,
  output: 1200,
  price_per_kwh: 0.25,
  watts: 450,
  hardware_cost: 0,
  api_input: 0.3,
  api_output: 1.2,
};

export function CostView({
  id,
  sizing,
  ranked,
  onOpen,
}: {
  id: string | null;
  sizing: Sizing;
  ranked: RankedModel[] | null;
  onOpen: (id: string, view?: View) => void;
}) {
  const [query, setQuery] = useState<CostQuery>(DEFAULTS);
  const [comparison, setComparison] = useState<CostComparison | null>(null);

  useEffect(() => {
    if (!id) return;
    let current = true;
    engine.cost(id, sizing, query).then((result) => {
      if (current) setComparison(result);
    });
    return () => {
      current = false;
    };
  }, [id, sizing, query]);

  const model = ranked?.find((entry) => entry.id === id);

  if (!id) {
    return (
      <Empty title="No model chosen">
        Pick one from Models to compare running it here against paying an API.
      </Empty>
    );
  }
  if (!comparison) return <Empty title="Working it out">One moment.</Empty>;

  const set = <K extends keyof CostQuery>(key: K, value: number) =>
    setQuery({ ...query, [key]: value });

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <Eyebrow>A month of</Eyebrow>
          <h1 className="mt-1.5 font-display text-[26px] font-semibold leading-none tracking-tight">
            {model?.name ?? id}
          </h1>
        </div>
        <button
          type="button"
          onClick={() => onOpen(id, "Plan")}
          className="rounded-lg border border-[var(--color-line)] px-3 py-1.5 text-[13px] hover:bg-[var(--color-raised)]"
        >
          Back to the plan
        </button>
      </div>

      <Card className="px-8 pb-7 pt-6">
        <Verdict comparison={comparison} />

        <div className="mt-7 grid gap-8 sm:grid-cols-2">
          <Side
            title="Here"
            total={comparison.local_total}
            perMtok={comparison.local_per_mtok}
            rows={[
              ["Electricity", comparison.local_energy],
              ["Hardware, amortised", comparison.local_amortisation],
            ]}
            emphasis={comparison.verdict.verdict === "local_cheaper"}
          />
          <Side
            title="An API"
            total={comparison.api_total}
            perMtok={comparison.api_per_mtok}
            rows={[
              ["Input tokens", (query.requests * query.input * query.api_input) / 1e6],
              ["Output tokens", (query.requests * query.output * query.api_output) / 1e6],
            ]}
            emphasis={comparison.verdict.verdict === "api_cheaper"}
          />
        </div>

        <dl className="mt-7 border-t border-[var(--color-line-soft)] pt-5">
          <Spec label="Generation time per month">
            <Figure
              value={comparison.compute_hours_per_month.toFixed(1)}
              unit="hours"
              confidence={model?.fit.confidence}
            />
          </Spec>
          {comparison.breakeven_requests_per_month !== null && (
            <Spec label="Break-even volume">
              <span className="figure text-[13px]">
                {Math.round(
                  comparison.breakeven_requests_per_month,
                ).toLocaleString()}{" "}
                requests a month
              </span>
            </Spec>
          )}
        </dl>

        {!comparison.prefill_included && (
          <p className="mt-5 rounded-lg bg-[var(--color-brass-wash)] px-4 py-3 text-[12px] leading-relaxed text-[var(--color-ink-dim)]">
            Prompt processing is not counted. Nothing has measured this
            machine's compute throughput, so the time spent reading your prompt
            is unknown — and unknown is reported as unknown rather than charged
            at zero. The local figure is therefore a floor, not an estimate.
          </p>
        )}
      </Card>

      <Card className="px-6 py-5">
        <Eyebrow>The assumptions</Eyebrow>
        <p className="mt-2 text-[12px] text-[var(--color-ink-faint)]">
          Change any of these and the comparison follows.
        </p>
        <div className="mt-4 grid gap-x-8 gap-y-4 sm:grid-cols-2">
          <Field
            label="Requests a month"
            value={query.requests}
            onChange={(v) => set("requests", v)}
          />
          <Field
            label="Prompt tokens each"
            value={query.input}
            onChange={(v) => set("input", v)}
          />
          <Field
            label="Generated tokens each"
            value={query.output}
            onChange={(v) => set("output", v)}
          />
          <Field
            label="Whole-system watts while generating"
            value={query.watts}
            onChange={(v) => set("watts", v)}
          />
          <Field
            label="Electricity per kWh"
            value={query.price_per_kwh}
            step={0.01}
            onChange={(v) => set("price_per_kwh", v)}
          />
          <Field
            label="Hardware to amortise (0 if owned)"
            value={query.hardware_cost}
            onChange={(v) => set("hardware_cost", v)}
          />
          <Field
            label="API, per million input tokens"
            value={query.api_input}
            step={0.01}
            onChange={(v) => set("api_input", v)}
          />
          <Field
            label="API, per million output tokens"
            value={query.api_output}
            step={0.01}
            onChange={(v) => set("api_output", v)}
          />
        </div>
      </Card>
    </div>
  );
}

function Verdict({ comparison }: { comparison: CostComparison }) {
  const { verdict } = comparison;
  if (verdict.verdict === "too_close_to_call") {
    return (
      <>
        <p className="font-display text-[24px] font-semibold leading-tight tracking-tight">
          Within a few percent
        </p>
        <p className="mt-2 max-w-xl text-[13px] leading-relaxed text-[var(--color-ink-dim)]">
          At this volume the two cost about the same, so the decision belongs on
          other grounds — whether the data may leave the machine, whether it
          must work offline, whether a rate limit would matter.
        </p>
      </>
    );
  }
  const local = verdict.verdict === "local_cheaper";
  return (
    <>
      <p className="font-display text-[24px] font-semibold leading-tight tracking-tight">
        {local ? "Running it here" : "Paying the API"} costs{" "}
        <span style={{ color: "var(--color-brass)" }}>
          {verdict.by_percent.toFixed(0)}% less
        </span>
      </p>
      <p className="mt-2 max-w-xl text-[13px] leading-relaxed text-[var(--color-ink-dim)]">
        {local
          ? `${fmt.money(comparison.local_total)} a month against ${fmt.money(
              comparison.api_total,
            )}, at this volume and these prices.`
          : `${fmt.money(comparison.api_total)} a month against ${fmt.money(
              comparison.local_total,
            )} in electricity and amortisation.`}
      </p>
    </>
  );
}

function Side({
  title,
  total,
  perMtok,
  rows,
  emphasis,
}: {
  title: string;
  total: number;
  perMtok: number;
  rows: [string, number][];
  emphasis: boolean;
}) {
  return (
    <div>
      <p className="eyebrow">{title}</p>
      <p
        className="figure mt-2 text-[32px] font-medium leading-none"
        style={emphasis ? { color: "var(--color-brass)" } : undefined}
      >
        {fmt.money(total)}
      </p>
      <p className="mt-1.5 text-[12px] text-[var(--color-ink-faint)]">
        a month · {fmt.money(perMtok)} per million tokens
      </p>
      <dl className="mt-3">
        {rows.map(([label, value]) => (
          <Spec key={label} label={label}>
            <span className="figure text-[13px] text-[var(--color-ink-dim)]">
              {fmt.money(value)}
            </span>
          </Spec>
        ))}
      </dl>
    </div>
  );
}

function Field({
  label,
  value,
  onChange,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: number;
}) {
  return (
    <label className="flex items-center justify-between gap-4">
      <span className="text-[13px] text-[var(--color-ink-dim)]">{label}</span>
      <input
        type="number"
        min={0}
        step={step}
        value={value}
        onChange={(event) => {
          const next = Number(event.target.value);
          if (Number.isFinite(next) && next >= 0) onChange(next);
        }}
        className="figure w-28 select-text rounded-lg border border-[var(--color-line)] bg-[var(--color-ground)] px-2.5 py-1 text-right text-[13px]"
      />
    </label>
  );
}
