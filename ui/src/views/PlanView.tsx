/**
 * Everything about running one model here.
 *
 * The memory column at full size, its bands named and explained, and beside it
 * the two curves that are the honest answer to "how fast is it": memory rising
 * with context, speed falling. A single tokens-per-second figure describes a
 * situation nobody is in.
 */
import { useEffect, useState } from "react";
import * as engine from "../engine";
import type { Launch, Plan, RankedModel, Sizing } from "../engine";
import type { View } from "../App";
import * as fmt from "../format";
import { MemoryColumn } from "../components/MemoryColumn";
import { Card, Empty, Eyebrow, Figure, Spec, VerdictPill } from "../components/ui";
import {
  Explain,
  quantExplanation,
  speedExplanation,
} from "../components/Explain";
import { Download, buildKey, useDownloads } from "../components/Download";
import { LaunchCard } from "../components/Launch";
import { runModeLabel } from "./ModelsView";

export function PlanView({
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
  const [plan, setPlan] = useState<Plan | null>(null);
  const [missing, setMissing] = useState(false);
  const [launch, setLaunch] = useState<Launch | null>(null);
  // Bumped when the download directory changes, because the launch names
  // the path and a command naming the old one would not run.
  const [destinationVersion, setDestinationVersion] = useState(0);
  const downloads = useDownloads();

  useEffect(() => {
    if (!id) return;
    let current = true;
    setMissing(false);
    engine.plan(id, sizing).then((result) => {
      if (!current) return;
      setPlan(result);
      setMissing(result === null);
    });
    return () => {
      current = false;
    };
  }, [id, sizing]);

  useEffect(() => {
    if (!id) return;
    let current = true;
    engine
      .launch(id, sizing)
      .then((result) => {
        if (current) setLaunch(result);
      })
      .catch(() => current && setLaunch(null));
    return () => {
      current = false;
    };
  }, [id, sizing, destinationVersion]);

  if (!id) {
    return (
      <Empty title="No model chosen">
        Pick one from Models and its full breakdown appears here.
      </Empty>
    );
  }
  if (missing) {
    return (
      <Empty title="That model is not in the catalog">
        It may have been dropped by a rebuild. Choose another from Models.
      </Empty>
    );
  }
  if (!plan) return <Empty title="Working it out">One moment.</Empty>;

  const { fit } = plan;
  const spare = Math.max(0, plan.pool_bytes - fit.memory.required);
  const chosen = plan.builds.find((build) => build.chosen);
  // Until the engine has said, the runtime is assumed to run the file: the
  // download control appears a moment before the launch card rather than the
  // whole section flickering in.
  const runsGguf = launch?.runs_gguf ?? true;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <Eyebrow>{plan.family}</Eyebrow>
          <h1 className="mt-1.5 font-display text-[26px] font-semibold leading-none tracking-tight">
            {plan.name}
          </h1>
          <p className="mt-2 select-text text-[12px] text-[var(--color-ink-faint)]">
            {plan.id}
            {plan.license && ` · ${plan.license}`}
            {plan.released && ` · released ${plan.released}`}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <VerdictPill {...fmt.verdict(fit.verdict)} />
          <button
            type="button"
            onClick={() => onOpen(plan.id, "Cost")}
            className="rounded-lg border border-[var(--color-line)] px-3 py-1.5 text-[13px] hover:bg-[var(--color-raised)]"
          >
            What it costs
          </button>
        </div>
      </div>

      <ModelPicker ranked={ranked} current={plan.id} onOpen={onOpen} />

      <Card className="px-8 pb-7 pt-6">
        <div className="flex items-baseline justify-between gap-4">
          <Eyebrow>Where the memory goes</Eyebrow>
          <span className="text-[12px] text-[var(--color-ink-faint)]">
            {fit.pool.label} · {fmt.bytes(plan.pool_bytes)} usable
          </span>
        </div>

        <div className="mt-5">
          <MemoryColumn
            memory={fit.memory}
            poolBytes={plan.pool_bytes}
            height={34}
            legend
          />
        </div>

        <p className="mt-6 border-t border-[var(--color-line-soft)] pt-5 text-[13px] leading-relaxed text-[var(--color-ink-dim)]">
          At {fmt.tokens(sizing.context)} context
          {sizing.parallel > 1 && ` and ${sizing.parallel} concurrent sequences`},{" "}
          {plan.name} as {fit.quant} needs{" "}
          <span className="figure text-[var(--color-ink)]">
            {fmt.bytes(fit.memory.required)}
          </span>{" "}
          of the {fmt.bytes(plan.pool_bytes)} available:{" "}
          {fmt.percent(fit.utilisation)} of the pool, leaving {fmt.bytes(spare)}.{" "}
          {fit.memory.weights_measured
            ? "The weight figure is the published file, measured."
            : "No published file exists at this format, so the weight figure is computed from the model's tensor shapes."}
        </p>
      </Card>

      {/* Directly under the accounting that justified this build, because
          the decision and the action are the same moment. The action changes
          with the runtime: a download for the three that run the file, and
          for the two that do not, no download and the command that works. */}
      {chosen && runsGguf && (
        <Download
          id={plan.id}
          quant={chosen.quant}
          bytes={chosen.bytes}
          progress={downloads[buildKey(plan.id, chosen.quant)]}
          builds={plan.builds}
          runtime={sizing.runtime}
          onDestinationChanged={() => setDestinationVersion((v) => v + 1)}
        />
      )}
      {launch && <LaunchCard launch={launch} id={plan.id} sizing={sizing} />}

      <div className="grid gap-6 lg:grid-cols-[1.35fr_1fr]">
        <Card className="px-6 pb-6 pt-5">
          <Eyebrow>Across context lengths</Eyebrow>
          <ContextCurve plan={plan} at={sizing.context} />
        </Card>

        <div className="flex flex-col gap-6">
          <Card className="px-6 py-5">
            <Eyebrow>The placement</Eyebrow>
            <dl className="mt-3">
              <Spec label="Runs on">
                <span className="text-[13px]">{runModeLabel(fit.run_mode)}</span>
              </Spec>
              <Spec label="Weight format">
                <Explain
                  className="figure text-[13px]"
                  {...quantExplanation(fit.quant, fit.bits_per_weight)}
                >
                  {fit.quant} · {fit.bits_per_weight.toFixed(2)} bpw
                </Explain>
              </Spec>
              <Spec label="Generation">
                <Explain underline={false} {...speedExplanation(fit.decode_tps)}>
                  <Figure
                    value={fmt.tps(fit.decode_tps)}
                    unit="tok/s"
                    confidence={fit.confidence}
                  />
                </Explain>
              </Spec>
              <Spec label="Prompt processing">
                {fit.prefill_tps === null ? (
                  <Explain
                    className="text-[13px] text-[var(--color-ink-faint)]"
                    title="Not counted, rather than zero"
                    body="How fast this machine reads a prompt depends on its compute throughput, and nothing has measured that yet. Charging it at zero would flatter every estimate, so the time is left out and said to be left out."
                  >
                    not counted
                  </Explain>
                ) : (
                  <Figure
                    value={fmt.tps(fit.prefill_tps)}
                    unit="tok/s"
                    confidence={fit.confidence}
                  />
                )}
              </Spec>
              <Spec label="Cache format">
                <Explain
                  className="figure text-[13px]"
                  title="Attention cache format"
                  body="The conversation is kept in memory in this format. Compressing it costs almost no quality and can halve what a long context needs, which is why it is chosen separately from the weights."
                >
                  {fit.kv_quant.toUpperCase()}
                </Explain>
              </Spec>
              <Spec label="Longest context here">
                <span className="figure text-[13px]">
                  {fit.max_context ? fmt.tokens(fit.max_context) : "not known"}
                </span>
              </Spec>
            </dl>
          </Card>

          <Card className="px-6 py-5">
            <Eyebrow>Quality</Eyebrow>
            <div className="mt-3.5 flex items-end gap-4">
              <Figure value={fit.quality.toFixed(1)} size="large" />
              <span className="pb-1 text-[12px] text-[var(--color-ink-faint)]">
                of {fit.quality_full_precision.toFixed(1)} at full precision
              </span>
            </div>
            <p className="mt-3 text-[12px] leading-relaxed text-[var(--color-ink-faint)]">
              {fit.quant} costs {fit.degradation.toFixed(1)} points against the
              unquantized model.{" "}
              {fit.quality_coverage >= 0.999
                ? "Every benchmark this use case weighs is published for this model."
                : `${fmt.percent(
                    fit.quality_coverage,
                  )} of what this use case weighs is backed by published evaluations; the rest is inferred from the family.`}
            </p>
          </Card>
        </div>
      </div>

      {fit.notes.length > 0 && (
        <Card className="px-6 py-5">
          <Eyebrow>Worth changing</Eyebrow>
          <ul className="mt-3 flex flex-col gap-2.5">
            {fit.notes.map((note, index) => (
              <li
                key={`${note.note}-${index}`}
                className="flex gap-3 text-[13px] leading-relaxed text-[var(--color-ink-dim)]"
              >
                <span
                  className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-[var(--color-brass)]"
                  aria-hidden
                />
                {fmt.fitNote(note)}
              </li>
            ))}
          </ul>
        </Card>
      )}

      <Card className="px-6 py-5">
        <Eyebrow>Every published build</Eyebrow>
        <p className="mt-2 text-[12px] text-[var(--color-ink-faint)]">
          {runsGguf
            ? "The solver chose one. Any of them can be fetched instead: a smaller format if the machine is shared, a larger one if there is room."
            : `Published as GGUF, which ${
                launch ? fmt.hostLabel(launch.host) : "this runtime"
              } does not run. Listed for their sizes, not offered for download.`}
        </p>
        <ul className="mt-3">
          {plan.builds.map((build) => {
            const state = downloads[buildKey(plan.id, build.quant)];
            return (
              <li
                key={build.quant}
                className="flex items-center gap-4 border-b border-[var(--color-line-soft)] py-2.5 last:border-0"
              >
                <span
                  className={`figure w-20 shrink-0 text-[13px] ${
                    build.chosen ? "text-[var(--color-brass)]" : ""
                  }`}
                >
                  {build.quant}
                </span>
                <span className="figure w-20 shrink-0 text-[13px] text-[var(--color-ink-dim)]">
                  {fmt.bytes(build.bytes)}
                </span>
                {build.chosen && (
                  <span className="shrink-0 text-[11px] text-[var(--color-brass)]">
                    chosen
                  </span>
                )}
                <span className="ml-auto shrink-0 text-[12px] text-[var(--color-ink-faint)]">
                  {buildStatus(state)}
                </span>
                {!build.chosen && !state && runsGguf && (
                  <button
                    type="button"
                    onClick={() =>
                      void engine.downloadBuild(plan.id, build.quant, sizing.runtime)
                    }
                    className="shrink-0 rounded-lg border border-[var(--color-line)] px-2.5 py-1 text-[12px] transition-colors hover:bg-[var(--color-raised)]"
                  >
                    Download
                  </button>
                )}
              </li>
            );
          })}
        </ul>
        {plan.builds.length === 0 && (
          <p className="mt-2 text-[13px] text-[var(--color-ink-faint)]">
            No GGUF build is published for this model, so every size above is
            computed from its tensor shapes rather than measured.
          </p>
        )}
      </Card>
    </div>
  );
}

/**
 * Memory and speed against context, on one pair of axes.
 *
 * Drawn rather than charted with a library: two series on a shared x-axis is
 * a hundred lines of SVG, and a charting dependency would be most of the
 * bundle for it.
 */
function ContextCurve({ plan, at }: { plan: Plan; at: number }) {
  const points = plan.curve;
  if (points.length < 2) {
    return (
      <p className="mt-3 text-[13px] text-[var(--color-ink-faint)]">
        This model has one context length, so there is no curve to draw.
      </p>
    );
  }

  const W = 460;
  const H = 170;
  const PAD = { top: 12, right: 8, bottom: 24, left: 8 };
  const maxRequired = Math.max(...points.map((p) => p.required));
  const maxTps = Math.max(...points.map((p) => p.decode_tps));
  // Scaled to the curve, not to the pool. A model using a fifth of a large
  // pool would otherwise plot as a flat line along the bottom, hiding the
  // shape this chart exists to show. Where the pool is close enough to matter
  // it is drawn as the ceiling; where it is not, it is stated in words
  // instead of squashing everything to make room for it.
  const axisMax = maxRequired * 1.15;
  const ceilingVisible = plan.pool_bytes <= axisMax;

  const x = (index: number) =>
    PAD.left + (index / (points.length - 1)) * (W - PAD.left - PAD.right);
  const yMem = (value: number) =>
    PAD.top + (1 - value / axisMax) * (H - PAD.top - PAD.bottom);
  const yTps = (value: number) =>
    PAD.top + (1 - value / maxTps) * (H - PAD.top - PAD.bottom);

  const line = (accessor: (p: (typeof points)[number]) => number, y: (v: number) => number) =>
    points.map((p, i) => `${i === 0 ? "M" : "L"}${x(i)},${y(accessor(p))}`).join(" ");

  const area =
    `M${x(0)},${yMem(points[0]!.required)} ` +
    points.map((p, i) => `L${x(i)},${yMem(p.required)}`).join(" ") +
    ` L${x(points.length - 1)},${H - PAD.bottom} L${x(0)},${H - PAD.bottom} Z`;

  const current = points.findIndex((p) => p.context >= at);
  const poolY = yMem(plan.pool_bytes);
  const worst = points.at(-1)!;

  return (
    <>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="mt-3 w-full"
        role="img"
        aria-label={`Memory rises from ${fmt.bytes(
          points[0]!.required,
        )} to ${fmt.bytes(points.at(-1)!.required)} and speed falls from ${fmt.tps(
          points[0]!.decode_tps,
        )} to ${fmt.tps(points.at(-1)!.decode_tps)} tokens per second`}
      >
        {/* The pool ceiling: the line the memory curve must not cross. */}
        {ceilingVisible && (
          <>
            <line
              x1={PAD.left}
              x2={W - PAD.right}
              y1={poolY}
              y2={poolY}
              stroke="var(--color-over)"
              strokeWidth="1"
              strokeDasharray="3 3"
              opacity="0.7"
            />
            <text
              x={W - PAD.right}
              y={poolY - 5}
              textAnchor="end"
              className="figure"
              fontSize="9"
              fill="var(--color-over)"
            >
              pool {fmt.bytes(plan.pool_bytes)}
            </text>
          </>
        )}

        <path d={area} fill="var(--color-band-weights)" opacity="0.1" />
        <path
          d={line((p) => p.required, yMem)}
          fill="none"
          stroke="var(--color-band-weights)"
          strokeWidth="1.75"
        />
        <path
          d={line((p) => p.decode_tps, yTps)}
          fill="none"
          stroke="var(--color-band-cache)"
          strokeWidth="1.75"
          strokeDasharray="4 2.5"
        />

        {current >= 0 && (
          <line
            x1={x(current)}
            x2={x(current)}
            y1={PAD.top}
            y2={H - PAD.bottom}
            stroke="var(--color-ink-faint)"
            strokeWidth="1"
            opacity="0.4"
          />
        )}

        {points.map((p, i) =>
          i === 0 || i === points.length - 1 || p.context === at ? (
            <text
              key={p.context}
              x={x(i)}
              y={H - 8}
              textAnchor={i === 0 ? "start" : i === points.length - 1 ? "end" : "middle"}
              className="figure"
              fontSize="9"
              fill="var(--color-ink-faint)"
            >
              {fmt.tokens(p.context)}
            </text>
          ) : null,
        )}
      </svg>

      <div className="mt-1 flex flex-wrap items-baseline gap-x-5 gap-y-1 text-[11px] text-[var(--color-ink-faint)]">
        <span className="flex items-center gap-1.5">
          <span
            className="h-0.5 w-4"
            style={{ background: "var(--color-band-weights)" }}
            aria-hidden
          />
          memory {fmt.bytes(points[0]!.required)} → {fmt.bytes(points.at(-1)!.required)}
        </span>
        <span className="flex items-center gap-1.5">
          <span
            className="h-0.5 w-4"
            style={{
              backgroundImage:
                "repeating-linear-gradient(90deg, var(--color-band-cache) 0 4px, transparent 4px 7px)",
            }}
            aria-hidden
          />
          speed {fmt.tps(points[0]!.decode_tps)} → {fmt.tps(points.at(-1)!.decode_tps)} tok/s
        </span>
      </div>
      <p className="mt-3 text-[12px] leading-relaxed text-[var(--color-ink-faint)]">
        Speed falls as the conversation grows because the attention cache is
        read for every token generated, and it gets bigger with every token
        stored.{" "}
        {ceilingVisible
          ? "The dashed red line is the end of the pool."
          : `Even at ${fmt.tokens(worst.context)} this stays well inside the ${fmt.bytes(
              plan.pool_bytes,
            )} pool, so the ceiling is off this chart.`}
      </p>
    </>
  );
}

/** A build's transfer, in a few words for a table that has no room. */
function buildStatus(state: engine.Progress | undefined): string {
  switch (state?.state) {
    case "starting":
      return "starting…";
    case "running":
      return state.total > 0
        ? `${Math.round((state.received / state.total) * 100)}%`
        : "downloading";
    case "paused":
      return "paused";
    case "done":
      return "downloaded";
    case "failed":
      return "failed";
    default:
      return "";
  }
}

/** Move between models without going back to the list. */
function ModelPicker({
  ranked,
  current,
  onOpen,
}: {
  ranked: RankedModel[] | null;
  current: string;
  onOpen: (id: string, view?: View) => void;
}) {
  if (!ranked || ranked.length === 0) return null;
  return (
    <label className="flex items-center gap-3">
      <span className="text-[12px] text-[var(--color-ink-dim)]">Model</span>
      <select
        value={current}
        onChange={(event) => onOpen(event.target.value, "Plan")}
        className="max-w-md flex-1 rounded-lg border border-[var(--color-line)] bg-[var(--color-surface)] px-2.5 py-1.5 text-[13px]"
      >
        {ranked.map((model, index) => (
          <option key={model.id} value={model.id}>
            {index + 1}. {model.name} · {model.fit.quant},{" "}
            {fmt.bytes(model.fit.memory.required)}, {fmt.tps(model.fit.decode_tps)} tok/s
          </option>
        ))}
      </select>
    </label>
  );
}
