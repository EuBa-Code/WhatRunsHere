/**
 * The memory column: a model's bytes drawn against the pool they must fit in.
 *
 * This is the one thing WhatLLM knows that a parameter count cannot express,
 * so it is the one thing the window is built around. It appears three times at
 * three sizes (as a row in the model list, as the machine's own pool, and
 * full size in a plan), and it is the same object each time, so a shape learnt
 * in one place is readable in the others.
 *
 * The bands are ordered by how well the engine knows them. Weights come from a
 * published file wherever one exists, so they are brass, the accent colour.
 * Overhead is the term it estimates, so it is nearly the ground. Headroom is
 * not drawn at all: it is deliberately unused memory, and colouring it would
 * make room left over look like room consumed.
 */
import type { Footprint } from "../engine";
import * as fmt from "../format";

/** One band of the column: what it is, how many bytes, what colour. */
interface Band {
  key: string;
  label: string;
  bytes: number;
  colour: string;
  /** What the band is, for someone who has not met the term. */
  detail: string;
}

export function bands(memory: Footprint): Band[] {
  return [
    {
      key: "weights",
      label: "Weights",
      bytes: memory.weights,
      colour: "var(--color-band-weights)",
      detail: memory.weights_measured
        ? "The published file, measured. Not an estimate of it."
        : "Computed from the model's tensor shapes. No published file to measure.",
    },
    {
      key: "kv_cache",
      label: "Attention cache",
      bytes: memory.kv_cache,
      colour: "var(--color-band-cache)",
      detail:
        "Grows with context and with every concurrent sequence. The term that decides how long a conversation can get.",
    },
    {
      key: "activations",
      label: "Activations",
      bytes: memory.activations,
      colour: "var(--color-band-activations)",
      detail:
        memory.attention_scores > 0
          ? `Includes ${fmt.bytes(
              memory.attention_scores,
            )} of attention score matrix, which flash attention would remove entirely.`
          : "The graph working set. Flash attention keeps the score matrix out of memory.",
    },
    {
      key: "runtime_overhead",
      label: "Runtime",
      bytes: memory.runtime_overhead,
      colour: "var(--color-band-overhead)",
      detail: "What the runtime itself holds, before any of the model.",
    },
  ].filter((band) => band.bytes > 0);
}

interface Props {
  memory: Footprint;
  /** Bytes the pool can supply. The column is drawn against this. */
  poolBytes: number;
  /** Height of the bar in pixels. */
  height?: number;
  /** Show the band names and byte figures beneath. */
  legend?: boolean;
  /** A previous footprint to mark, so a change is visible as a change. */
  compareBytes?: number;
  /** True while a probe is running, which the bar shows rather than states. */
  busy?: boolean;
}

export function MemoryColumn({
  memory,
  poolBytes,
  height = 12,
  legend = false,
  compareBytes,
  busy = false,
}: Props) {
  const parts = bands(memory);
  // Drawn against the pool, not against the model: a bar that always filled
  // its width would make every model look equally close to the limit, which
  // is the single thing this drawing exists to distinguish.
  const scale = poolBytes > 0 ? poolBytes : memory.required;
  const overflows = memory.required > poolBytes && poolBytes > 0;
  const width = (value: number) => `${Math.min(100, (value / scale) * 100)}%`;

  return (
    <div className="w-full">
      <div
        className={`relative flex w-full overflow-hidden rounded-full ${
          busy ? "sweeping" : ""
        }`}
        style={{ height, background: "var(--color-line-soft)" }}
        role="img"
        aria-label={`${fmt.bytes(memory.required)} of ${fmt.bytes(
          poolBytes,
        )} available${overflows ? ", which does not fit" : ""}`}
      >
        {parts.map((band) => (
          <div
            key={band.key}
            style={{ width: width(band.bytes), background: band.colour }}
            title={`${band.label} · ${fmt.bytes(band.bytes)}`}
          />
        ))}
        {/* Headroom, hatched rather than filled: memory the runtime keeps
            free on purpose is neither used nor available. */}
        {memory.headroom > 0 && !overflows && (
          <div
            style={{
              width: width(memory.headroom),
              backgroundImage:
                "repeating-linear-gradient(45deg, var(--color-line) 0 2px, transparent 2px 5px)",
            }}
            title={`Headroom · ${fmt.bytes(memory.headroom)} left free by the allocator`}
          />
        )}
        {overflows && (
          <div
            className="absolute inset-y-0 right-0 w-1"
            style={{ background: "var(--color-over)" }}
            title="Past the end of the pool"
          />
        )}
        {compareBytes !== undefined && compareBytes > 0 && compareBytes < scale && (
          <div
            className="absolute inset-y-0 w-px"
            style={{
              left: width(compareBytes),
              background: "var(--color-ink)",
              opacity: 0.55,
            }}
            title={`Was ${fmt.bytes(compareBytes)}`}
          />
        )}
      </div>

      {legend && (
        <dl className="mt-4 grid gap-x-6 gap-y-3 sm:grid-cols-2">
          {parts.map((band) => (
            <div key={band.key} className="flex gap-3">
              <span
                className="mt-[5px] h-2.5 w-2.5 shrink-0 rounded-[3px]"
                style={{ background: band.colour }}
                aria-hidden
              />
              <div className="min-w-0">
                <dt className="flex items-baseline justify-between gap-3">
                  <span className="text-[13px] font-medium">{band.label}</span>
                  <span className="figure text-[13px] tabular-nums text-[var(--color-ink-dim)]">
                    {fmt.bytes(band.bytes)}
                  </span>
                </dt>
                <dd className="mt-0.5 text-[12px] leading-snug text-[var(--color-ink-faint)]">
                  {band.detail}
                </dd>
              </div>
            </div>
          ))}
          <div className="flex gap-3">
            <span
              className="mt-[5px] h-2.5 w-2.5 shrink-0 rounded-[3px]"
              style={{
                backgroundImage:
                  "repeating-linear-gradient(45deg, var(--color-line) 0 2px, transparent 2px 4px)",
                border: "1px solid var(--color-line)",
              }}
              aria-hidden
            />
            <div className="min-w-0">
              <dt className="flex items-baseline justify-between gap-3">
                <span className="text-[13px] font-medium">Headroom</span>
                <span className="figure text-[13px] tabular-nums text-[var(--color-ink-dim)]">
                  {fmt.bytes(memory.headroom)}
                </span>
              </dt>
              <dd className="mt-0.5 text-[12px] leading-snug text-[var(--color-ink-faint)]">
                Left free on purpose. An allocator that fills a pool exactly
                fails to allocate the next thing it needs.
              </dd>
            </div>
          </div>
        </dl>
      )}
    </div>
  );
}
