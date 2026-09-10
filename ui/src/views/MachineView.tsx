/**
 * What this machine is, and what measured it.
 *
 * The hero is the machine's memory pool with the best-fitting model drawn
 * inside it at true proportion. It is the product's whole claim in one
 * picture: here is what you have, here is what goes in it, here is what is
 * left. Everything below it is evidence for that picture.
 */
import type { Machine, RankedModel } from "../engine";
import type { View } from "../App";
import * as fmt from "../format";
import { MemoryColumn } from "../components/MemoryColumn";
import { Card, Empty, Eyebrow, Figure, Spec, VerdictPill } from "../components/ui";

export function MachineView({
  machine,
  measuring,
  onMeasure,
  best,
  onOpen,
}: {
  machine: Machine | null;
  measuring: boolean;
  onMeasure: () => void;
  best: RankedModel | null;
  onOpen: (id: string, view?: View) => void;
}) {
  if (!machine) {
    return <Empty title="Reading the machine">This takes a moment.</Empty>;
  }

  const { system, notes } = machine.detection;
  const pool = machine.pools[0];
  const measured = machine.measurement;

  return (
    <div className="flex flex-col gap-6">
      <Card className="px-8 pb-8 pt-7">
        <Eyebrow>This machine</Eyebrow>
        <h1 className="mt-2 font-display text-[28px] font-semibold leading-tight tracking-tight">
          {system.cpu.brand}
        </h1>
        <p className="mt-1 text-[13px] text-[var(--color-ink-dim)]">
          {system.os} · {system.cpu.physical_cores} cores,{" "}
          {system.cpu.logical_cores} threads
          {system.accelerators.length > 0 &&
            ` · ${system.accelerators.map((a) => a.name).join(", ")}`}
        </p>

        {pool && (
          <div className="mt-7">
            <div className="mb-2.5 flex items-baseline justify-between gap-4">
              <span className="text-[13px] text-[var(--color-ink-dim)]">
                {pool.label} · {pool.kind === "unified" ? "unified memory" : pool.kind}
              </span>
              <span className="figure text-[13px] text-[var(--color-ink-dim)]">
                {fmt.bytes(pool.usable_bytes)} usable
              </span>
            </div>

            {best ? (
              <>
                <MemoryColumn
                  memory={best.fit.memory}
                  poolBytes={pool.usable_bytes}
                  height={30}
                />
                <div className="mt-3.5 flex flex-wrap items-baseline gap-x-3 gap-y-1.5">
                  <button
                    type="button"
                    onClick={() => onOpen(best.id)}
                    className="font-display text-[15px] font-semibold hover:text-[var(--color-brass)]"
                  >
                    {best.name}
                  </button>
                  <span className="figure text-[12px] text-[var(--color-ink-faint)]">
                    {best.fit.quant}
                  </span>
                  <VerdictPill {...fmt.verdict(best.fit.verdict)} />
                  <span className="ml-auto text-[13px] text-[var(--color-ink-dim)]">
                    <Figure
                      value={fmt.tps(best.fit.decode_tps)}
                      unit="tok/s"
                      confidence={best.fit.confidence}
                    />
                  </span>
                </div>
                <p className="mt-2.5 text-[13px] leading-relaxed text-[var(--color-ink-dim)]">
                  The best fit takes {fmt.bytes(best.fit.memory.required)} of the{" "}
                  {fmt.bytes(pool.usable_bytes)} this machine can offer, leaving{" "}
                  {fmt.bytes(
                    Math.max(0, pool.usable_bytes - best.fit.memory.required),
                  )}
                  .{" "}
                  <button
                    type="button"
                    onClick={() => onOpen(best.id, "Models")}
                    className="underline decoration-[var(--color-line)] underline-offset-2 hover:text-[var(--color-ink)]"
                  >
                    See the rest
                  </button>
                </p>
              </>
            ) : (
              <MemoryColumn
                memory={{
                  weights: 0,
                  weights_measured: false,
                  kv_cache: 0,
                  activations: 0,
                  attention_scores: 0,
                  runtime_overhead: 0,
                  headroom: 0,
                  resident: 0,
                  required: 0,
                }}
                poolBytes={pool.usable_bytes}
                height={30}
              />
            )}
          </div>
        )}
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="px-6 py-5">
          <Eyebrow>Measured</Eyebrow>
          <div className="mt-4 flex items-end justify-between gap-6">
            <div>
              <Figure
                value={fmt.bandwidth(machine.calibration.host.bandwidth_bytes_per_s)}
                confidence={machine.calibration.source}
                size="large"
              />
              <p className="mt-1.5 text-[12px] text-[var(--color-ink-faint)]">
                {measured
                  ? `Streaming bandwidth, measured ${fmt.since(measured.measured_at)}`
                  : "Assumed. Nothing has measured this machine."}
              </p>
            </div>
            <button
              type="button"
              onClick={onMeasure}
              disabled={measuring}
              className="shrink-0 rounded-lg bg-[var(--color-brass)] px-3.5 py-1.5 text-[13px] font-medium text-[#17140d] transition-opacity hover:opacity-90 disabled:opacity-50"
            >
              {measuring ? "Measuring…" : measured ? "Measure again" : "Measure"}
            </button>
          </div>
          <p className="mt-4 text-[12px] leading-relaxed text-[var(--color-ink-faint)]">
            Generation is bound by how fast weights can be read, not by how fast
            the processor is. A measurement here takes about a second and
            replaces the estimate behind every speed on the other views.
          </p>
          {measured && (
            <dl className="mt-4">
              <Spec label="One thread">
                <Figure
                  value={fmt.bandwidth(measured.single_thread_bytes_per_s)}
                  confidence="calibrated"
                />
              </Spec>
              <Spec label="Bought by parallelism">
                <span className="figure text-[13px]">
                  {(
                    measured.host_bytes_per_s / measured.single_thread_bytes_per_s
                  ).toFixed(1)}
                  ×
                </span>
              </Spec>
              <Spec label="Per-token overhead">
                <span className="figure text-[13px]">
                  {machine.calibration.overhead_ms_per_token.toFixed(2)} ms
                </span>
              </Spec>
            </dl>
          )}
        </Card>

        <Card className="px-6 py-5">
          <Eyebrow>Memory</Eyebrow>
          <dl className="mt-3">
            <Spec label="Installed">
              <Figure value={fmt.bytes(system.memory.total_bytes)} />
            </Spec>
            <Spec label="Free now">
              <Figure value={fmt.bytes(system.memory.available_bytes)} />
            </Spec>
            {system.memory.uma_carveout_bytes !== undefined && (
              <Spec label="Firmware carveout">
                <Figure value={fmt.bytes(system.memory.uma_carveout_bytes)} />
              </Spec>
            )}
            {machine.pools.map((entry) => (
              <Spec key={entry.label} label={`${entry.label}, usable`}>
                <Figure value={fmt.bytes(entry.usable_bytes)} />
              </Spec>
            ))}
            <Spec label="Catalog">
              <span className="text-[13px] text-[var(--color-ink-dim)]">
                {machine.catalog_size} models · {machine.catalog_generated}
              </span>
            </Spec>
          </dl>
        </Card>
      </div>

      {notes.length > 0 && (
        <Card className="px-6 py-5">
          <Eyebrow>What detection could not settle</Eyebrow>
          <ul className="mt-3 flex flex-col gap-2.5">
            {notes.map((note, index) => (
              <li
                key={`${note.note}-${index}`}
                className="flex gap-3 text-[13px] leading-relaxed text-[var(--color-ink-dim)]"
              >
                <span
                  className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-[var(--color-ink-faint)]"
                  aria-hidden
                />
                {detectionNote(note)}
              </li>
            ))}
          </ul>
        </Card>
      )}
    </div>
  );
}

/** The engine's caveats, in words. */
function detectionNote(note: Record<string, unknown> & { note: string }): string {
  const s = (key: string) => String(note[key]);
  const n = (key: string) => Number(note[key]);
  switch (note.note) {
    case "ignored_virtual_adapter":
      return `${s("name")} was ignored — a display, not something that computes.`;
    case "integrated_graphics":
      return note.claimed_vram_bytes
        ? `${s("name")} is integrated. The ${fmt.bytes(
            n("claimed_vram_bytes"),
          )} of "video memory" the system reports is an aperture, not memory it owns; its pool is system RAM.`
        : `${s("name")} is integrated; its pool is system RAM.`;
    case "legacy_integrated_graphics":
      return `${s(
        "name",
      )} is older than any usable compute back end, so the model will run on the processor. It reads the same memory either way, so nothing is lost by that.`;
    case "nvidia_unavailable":
      return "No NVIDIA driver found, which on most machines is simply the truth.";
    case "memory_size_unreadable":
      return `${s(
        "device",
      )}'s memory is only reported through a 32-bit field, which cannot hold the size of any card worth asking about. It was left out rather than sized against a ceiling value.`;
    case "bandwidth_unknown":
      return `Nothing has measured ${s("device")}'s bandwidth.`;
    case "firmware_memory_carveout":
      return `Firmware reserved ${fmt.bytes(
        n("bytes"),
      )} for graphics before the operating system started, so it reports only ${fmt.bytes(
        n("os_reported_bytes"),
      )}. That memory is installed and a model can use it; it has been counted back in.`;
    case "compute_memory_capped":
      return `${s("device")} shares the machine's ${fmt.bytes(
        n("pool_bytes"),
      )}, but the platform will not let a compute job hold more than ${fmt.bytes(
        n("bytes"),
      )} of it. Plans are sized against the smaller figure.`;
    case "platform_unsupported":
      return `No adapter enumeration for ${s("target")} yet.`;
    default:
      return note.note.replace(/_/g, " ");
  }
}
