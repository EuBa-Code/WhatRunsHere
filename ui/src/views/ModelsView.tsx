/**
 * The catalog, ranked for this machine.
 *
 * Every row carries the same memory column as the hero, at row height, so the
 * list can be read as a set of fills rather than a set of numbers: which
 * models sit comfortably, which are pressed against the end of the pool, which
 * do not fit at all. That comparison is the reason to have a list.
 */
import { useMemo, useState } from "react";
import type { Installed, InstalledModel, RankedModel, Sizing } from "../engine";
import type { View } from "../App";
import * as fmt from "../format";
import { MemoryColumn } from "../components/MemoryColumn";
import { Empty, Eyebrow, Figure, Meter, VerdictPill } from "../components/ui";
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
  installed,
  onOpen,
}: {
  ranked: RankedModel[] | null;
  sizing: Sizing;
  selectedId: string | null;
  /** What is already on the disk, or null until the engine has looked. */
  installed: Installed | null;
  onOpen: (id: string, view?: View) => void;
}) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<Filter>("all");

  // Catalog ids with a build somewhere on this machine, for the tag on the
  // ranked rows.
  const onDisk = useMemo(() => {
    const ids = new Set<string>();
    for (const file of installed?.files ?? []) {
      if (file.identity.kind === "catalog") ids.add(file.identity.id);
    }
    return ids;
  }, [installed]);

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
      {installed && <OnThisMachine installed={installed} onOpen={onOpen} />}

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
              onDisk={onDisk.has(model.id)}
              onOpen={onOpen}
            />
          ))}
        </ol>
      )}
    </div>
  );
}

/**
 * What is already here, before the catalog.
 *
 * A person who has used two runtimes has weights in two places and no list.
 * This is the list, with each file the catalog recognises solved for exactly
 * the build it is, so "which models do I have" comes with "and how would
 * each of them run here". A file the catalog does not know is named from its
 * own header and said to be unsizeable, not sized from a guess.
 */
function OnThisMachine({
  installed,
  onOpen,
}: {
  installed: Installed;
  onOpen: (id: string, view?: View) => void;
}) {
  const total = installed.files.reduce((sum, file) => sum + file.bytes, 0);
  // WhatLLM's own directory not existing yet says nothing about the person.
  const absent = installed.looked_in.filter(
    (place) => !place.found && place.provider !== "what_llm",
  );

  // Nothing found is said in one line, with where it was looked for, so an
  // empty machine reads as empty rather than as a feature that is not there.
  if (installed.files.length === 0) {
    return (
      <p className="text-[12px] text-[var(--color-ink-faint)]">
        Nothing on this machine yet. Looked in{" "}
        {installed.looked_in.map((place) => fmt.providerLabel(place.provider)).join(", ")}
        ; a model downloaded from Plan appears here, sized for this machine.
      </p>
    );
  }

  return (
    <section className="card px-5 pb-4 pt-4">
      <div className="flex items-baseline justify-between gap-4">
        <Eyebrow>Already on this machine</Eyebrow>
        <span className="figure text-[12px] text-[var(--color-ink-faint)]">
          {installed.files.length} {installed.files.length === 1 ? "file" : "files"} ·{" "}
          {fmt.bytes(total)}
        </span>
      </div>
      <ul className="mt-2 flex flex-col">
        {installed.files.map((file) => (
          <InstalledRow key={file.path} file={file} onOpen={onOpen} />
        ))}
      </ul>
      {absent.length > 0 && (
        <p className="mt-3 text-[11px] text-[var(--color-ink-faint)]">
          Not found: {absent.map((place) => fmt.providerLabel(place.provider)).join(", ")}.
          Looked in the folder each keeps by default.
        </p>
      )}
    </section>
  );
}

function InstalledRow({
  file,
  onOpen,
}: {
  file: InstalledModel;
  onOpen: (id: string, view?: View) => void;
}) {
  const { identity, fit } = file;
  const where = (
    <span
      className="block truncate text-[11px] text-[var(--color-ink-faint)]"
      title={file.path}
    >
      {fmt.providerLabel(file.provider)}
      {file.provider === "ollama" && ` · ${file.name}`}
      {" · "}
      {file.path}
    </span>
  );

  if (identity.kind === "catalog") {
    return (
      <li className="border-b border-[var(--color-line-soft)] py-2.5 last:border-0">
        <button
          type="button"
          onClick={() => onOpen(identity.id)}
          className="flex w-full items-center gap-4 text-left"
        >
          <span className="min-w-0 flex-1">
            <span className="flex items-baseline gap-2">
              <span className="font-display text-[14px] font-semibold tracking-tight">
                {identity.display_name}
              </span>
              <span className="figure text-[12px] text-[var(--color-ink-faint)]">
                {identity.quant} · {fmt.bytes(file.bytes)}
              </span>
            </span>
            {where}
          </span>
          {fit ? (
            <span className="flex shrink-0 items-center gap-4">
              <span className="figure text-[12px] text-[var(--color-ink-dim)]">
                {fmt.bytes(fit.memory.required)}
              </span>
              <Figure value={fmt.tps(fit.decode_tps)} unit="tok/s" confidence={fit.confidence} />
              <VerdictPill {...fmt.verdict(fit.verdict)} />
            </span>
          ) : (
            <span className="shrink-0 text-[12px] text-[var(--color-ink-faint)]">
              does not fit at this context
            </span>
          )}
        </button>
      </li>
    );
  }

  const described =
    identity.kind === "header"
      ? [identity.name, identity.architecture, identity.format]
          .filter((part): part is string => Boolean(part))
          .join(", ") +
        (identity.shard ? `, part ${identity.shard[0]} of ${identity.shard[1]}` : "")
      : identity.reason;

  return (
    <li className="flex items-center gap-4 border-b border-[var(--color-line-soft)] py-2.5 last:border-0">
      <span className="min-w-0 flex-1">
        <span className="flex items-baseline gap-2">
          <span className="font-display text-[14px] font-semibold tracking-tight">
            {file.name}
          </span>
          <span className="figure text-[12px] text-[var(--color-ink-faint)]">
            {fmt.bytes(file.bytes)}
          </span>
        </span>
        {where}
      </span>
      <span className="max-w-[45%] shrink-0 text-right text-[12px] text-[var(--color-ink-faint)]">
        {identity.kind === "header"
          ? `Not in the catalog (${described}), so it cannot be sized here.`
          : described}
      </span>
    </li>
  );
}

function Row({
  model,
  rank,
  selected,
  onDisk,
  onOpen,
}: {
  model: RankedModel;
  rank: number;
  selected: boolean;
  /** A build of this model is somewhere on this machine already. */
  onDisk: boolean;
  onOpen: (id: string, view?: View) => void;
}) {
  const { fit } = model;
  const runs = fit.verdict !== "does_not_fit";

  // The row opens a plan, and it also carries explanations that can be
  // hovered and focused. One cannot be nested inside the other (a button
  // inside a button is invalid, and a screen reader reads the outer label over
  // the inner one), so the row's own action is a button stretched behind the
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
          <span className="ml-auto flex shrink-0 items-center gap-3">
            {onDisk && (
              <span className="text-[11px] text-[var(--color-good)]">on this machine</span>
            )}
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
