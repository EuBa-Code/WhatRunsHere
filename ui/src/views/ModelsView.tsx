/**
 * The catalog, ranked for this machine.
 *
 * Every row carries the same memory column as the hero, at row height, so the
 * list can be read as a set of fills rather than a set of numbers: which
 * models sit comfortably, which are pressed against the end of the pool, which
 * do not fit at all. That comparison is the reason to have a list.
 */
import { useMemo, useState } from "react";
import type { RankedModel, Sizing } from "../engine";
import type { View } from "../App";
import * as fmt from "../format";
import { MemoryColumn } from "../components/MemoryColumn";
import { Empty, Figure, Meter, VerdictPill } from "../components/ui";
import {
  Explain,
  qualityExplanation,
  quantExplanation,
  speedExplanation,
} from "../components/Explain";

type Filter = "all" | "runs" | "sparse";

export function ModelsView({
  ranked,
  sizing,
  selectedId,
  onOpen,
}: {
  ranked: RankedModel[] | null;
  sizing: Sizing;
  selectedId: string | null;
  onOpen: (id: string, view?: View) => void;
}) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<Filter>("all");

  const shown = useMemo(() => {
    if (!ranked) return [];
    const needle = query.trim().toLowerCase();
    return ranked.filter((model) => {
      if (filter === "runs" && model.fit.verdict === "does_not_fit") return false;
      if (filter === "sparse" && !model.sparse) return false;
      if (!needle) return true;
      return (
        model.name.toLowerCase().includes(needle) ||
        model.family.toLowerCase().includes(needle) ||
        model.id.toLowerCase().includes(needle)
      );
    });
  }, [ranked, query, filter]);

  if (!ranked) {
    return <Empty title="Ranking the catalog">Solving every model for this machine.</Empty>;
  }

  const fits = ranked.filter((m) => m.fit.verdict !== "does_not_fit").length;

  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center gap-4">
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Filter by name or family"
          className="w-56 select-text rounded-lg border border-[var(--color-line)] bg-[var(--color-surface)] px-3 py-1.5 text-[13px] placeholder:text-[var(--color-ink-faint)]"
        />
        <div className="flex gap-0.5">
          {(
            [
              ["all", `All ${ranked.length}`],
              ["runs", `Runs here ${fits}`],
              ["sparse", "Mixture of experts"],
            ] as [Filter, string][]
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              onClick={() => setFilter(value)}
              className={`rounded-lg px-2.5 py-1 text-[12px] ${
                filter === value
                  ? "bg-[var(--color-raised)] text-[var(--color-ink)]"
                  : "text-[var(--color-ink-faint)] hover:text-[var(--color-ink-dim)]"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
        <p className="ml-auto text-[12px] text-[var(--color-ink-faint)]">
          At {fmt.tokens(sizing.context)} context
          {sizing.parallel > 1 && `, ${sizing.parallel} concurrent`}
        </p>
      </div>

      {shown.length === 0 ? (
        <Empty title="Nothing matches">
          No model in the catalog matches that filter. Clear it to see the rest.
        </Empty>
      ) : (
        <ol className="flex flex-col gap-2">
          {shown.map((model, index) => (
            <Row
              key={model.id}
              model={model}
              rank={index + 1}
              selected={model.id === selectedId}
              onOpen={onOpen}
            />
          ))}
        </ol>
      )}
    </div>
  );
}

function Row({
  model,
  rank,
  selected,
  onOpen,
}: {
  model: RankedModel;
  rank: number;
  selected: boolean;
  onOpen: (id: string, view?: View) => void;
}) {
  const { fit } = model;
  const runs = fit.verdict !== "does_not_fit";

  // The row opens a plan, and it also carries explanations that can be
  // hovered and focused. One cannot be nested inside the other — a button
  // inside a button is invalid, and a screen reader reads the outer label over
  // the inner one — so the row's own action is a button stretched behind the
  // content, and only the explanations take pointer events in front of it.
  return (
    <li
      className={`card group relative px-4 py-3 transition-colors focus-within:border-[var(--color-ink-faint)] hover:border-[var(--color-ink-faint)] ${
        selected ? "border-[var(--color-brass)]" : ""
      } ${runs ? "" : "opacity-55"}`}
    >
      <button
        type="button"
        onClick={() => onOpen(model.id)}
        className="absolute inset-0 z-0 rounded-[inherit]"
      >
        <span className="sr-only">
          Open the full plan for {model.name}
        </span>
      </button>
      <div className="pointer-events-none relative z-10">
        <div className="flex items-baseline gap-3">
          <span className="figure w-6 shrink-0 text-[11px] text-[var(--color-ink-faint)]">
            {rank}
          </span>
          <span className="font-display text-[15px] font-semibold tracking-tight">
            {model.name}
          </span>
          <span className="figure text-[12px] text-[var(--color-ink-faint)]">
            {fmt.params(model.parameters)}
            {model.sparse && (
              <>
                {" · "}
                <Explain
                  title="A mixture of experts"
                  body={`This model holds ${fmt.params(
                    model.parameters,
                  )} of weights but reads only ${fmt.params(
                    model.active_parameters,
                  )} of them for each token. It needs the memory of a large model and runs at close to the speed of a small one.`}
                >
                  {fmt.params(model.active_parameters)} active
                </Explain>
              </>
            )}
          </span>
          <span className="ml-auto shrink-0">
            <VerdictPill {...fmt.verdict(fit.verdict)} />
          </span>
        </div>

        <div className="mt-2.5 flex items-center gap-4">
          <div className="min-w-0 flex-1">
            <MemoryColumn
              memory={fit.memory}
              poolBytes={fit.pool.usable_bytes}
              height={8}
            />
          </div>
          <span className="figure w-20 shrink-0 text-right text-[12px] text-[var(--color-ink-dim)]">
            {fmt.bytes(fit.memory.required)}
          </span>
        </div>

        <div className="mt-2.5 flex items-baseline gap-5 text-[12px] text-[var(--color-ink-faint)]">
          <Explain
            className="figure"
            {...quantExplanation(fit.quant, fit.bits_per_weight)}
          >
            {fit.quant}
          </Explain>
          <Explain
            title="Attention cache format"
            body="The conversation is kept in memory in this format. Compressing it costs almost no quality and can halve what a long context needs, which is why it is chosen separately from the weights."
          >
            cache <span className="figure">{fit.kv_quant.toUpperCase()}</span>
          </Explain>
          <span>{runModeLabel(fit.run_mode)}</span>
          <span className="ml-auto flex items-baseline gap-4 whitespace-nowrap">
            <span className="flex items-baseline gap-1.5">
              <Explain {...qualityExplanation(fit.quality_rank)}>quality</Explain>
              <span className="figure text-[13px] text-[var(--color-ink)]">
                {Math.round(fit.quality)}
              </span>
              <span className="w-9">
                <Meter value={fit.quality} />
              </span>
            </span>
            <Explain underline={false} {...speedExplanation(fit.decode_tps)}>
              <Figure
                value={fmt.tps(fit.decode_tps)}
                unit="tok/s"
                confidence={fit.confidence}
              />
            </Explain>
          </span>
        </div>
      </div>
    </li>
  );
}

export function runModeLabel(mode: RankedModel["fit"]["run_mode"]): string {
  switch (mode.mode) {
    case "accelerated":
      return mode.devices > 1 ? `${mode.devices} accelerators` : "Accelerator";
    case "unified":
      return "Unified memory";
    case "offloaded":
      return `${mode.gpu_layers} of ${mode.total_layers} layers offloaded`;
    case "cpu":
      return "Processor";
  }
}
