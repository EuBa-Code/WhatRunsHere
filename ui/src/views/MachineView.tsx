/**
 * What this machine is, and what measured it.
 *
 * The hero is the machine's memory pool with the best-fitting model drawn
 * inside it at true proportion. It is the product's whole claim in one
 * picture: here is what you have, here is what goes in it, here is what is
 * left. Everything below it is evidence for that picture.
 *
 * On a machine nothing has measured, that picture is drawn from an assumption,
 * and the invitation to replace it comes first, before the machine, before
 * the model. It is one second of work and it sharpens every figure in the
 * application, which is the best thing this application does and a poor thing
 * to bury in a card.
 */
import type { Machine, RankedModel } from "../engine";
import type { View } from "../App";
import * as fmt from "../format";
import { MemoryColumn } from "../components/MemoryColumn";
import { Explain, speedExplanation } from "../components/Explain";
import { ShareButton } from "../components/ShareButton";
import { Card, Empty, Eyebrow, Figure, Spec, VerdictPill } from "../components/ui";

export function MachineView({
  machine,
  measuring,
  onMeasure,
  best,
  ranked,
  onOpen,
}: {
  machine: Machine | null;
  measuring: boolean;
  onMeasure: () => void;
  best: RankedModel | null;
  /** The whole ranking, so a shared result can carry more than one row. */
  ranked: RankedModel[] | null;
  onOpen: (id: string, view?: View) => void;
}) {
  if (!machine) {
    return <Empty title="Reading the machine">This takes a moment.</Empty>;
  }

  const { system, notes } = machine.detection;
  const pool = machine.pools[0];
  const poolName = machine.display.pools[0] ?? pool?.label ?? "";
  const measured = machine.measurement;

  return (
    <div className="flex flex-col gap-6">
      {!measured && (
        <FirstRun measuring={measuring} onMeasure={onMeasure} />
      )}

      <Card className="rising px-8 pb-8 pt-7" style={{ animationDelay: "40ms" }}>
        <Eyebrow>This machine</Eyebrow>
        <h1 className="mt-2 font-display text-[28px] font-semibold leading-tight tracking-tight">
          {machine.display.cpu}
        </h1>
        <p className="mt-1 text-[13px] text-[var(--color-ink-dim)]">
          {system.os} · {system.cpu.physical_cores} cores,{" "}
          {system.cpu.logical_cores} threads
          {machine.display.accelerators.length > 0 &&
            ` · ${machine.display.accelerators.join(", ")}`}
        </p>

        {pool && (
          <div className="mt-7">
            <div className="mb-2.5 flex items-baseline justify-between gap-4">
              <span className="text-[13px] text-[var(--color-ink-dim)]">
                {poolName} ·{" "}
                {pool.kind === "unified" ? (
                  <Explain
                    title="Unified memory"
                    body="This machine has one pool of memory that the processor and the graphics share, rather than a card with memory of its own. A model can use nearly all of it, and nothing has to be copied between the two."
                  >
                    unified memory
                  </Explain>
                ) : (
                  pool.kind
                )}
              </span>
              <span className="figure text-[13px] text-[var(--color-ink-dim)]">
                {fmt.bytes(pool.usable_bytes)}{" "}
                <Explain
                  title="Usable memory"
                  body="Installed memory less a reserve for the operating system and its caches. It is what a model can claim with the machine otherwise quiet, and deliberately not whatever happens to be free this second, so the answer does not change because a browser opened."
                >
                  usable
                </Explain>
              </span>
            </div>

            {best ? (
              <>
                <MemoryColumn
                  memory={best.fit.memory}
                  poolBytes={pool.usable_bytes}
                  height={30}
                  busy={measuring}
                />
                <div className="mt-3.5 flex flex-wrap items-baseline gap-x-3 gap-y-1.5">
                  <button
                    type="button"
                    onClick={() => onOpen(best.id)}
                    className="font-display text-[15px] font-semibold transition-colors hover:text-[var(--color-brass)]"
                  >
                    {best.name}
                  </button>
                  <span className="figure text-[12px] text-[var(--color-ink-faint)]">
                    {best.fit.quant}
                  </span>
                  <VerdictPill {...fmt.verdict(best.fit.verdict)} />
                  <span className="ml-auto text-[13px] text-[var(--color-ink-dim)]">
                    <Explain underline={false} {...speedExplanation(best.fit.decode_tps)}>
                      <Figure
                        value={fmt.tps(best.fit.decode_tps)}
                        unit="tok/s"
                        confidence={best.fit.confidence}
                      />
                    </Explain>
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
                    className="underline decoration-[var(--color-line)] underline-offset-2 transition-colors hover:text-[var(--color-ink)]"
                  >
                    See the rest
                  </button>
                </p>

                {ranked && ranked.length > 0 && (
                  <div className="mt-4 border-t border-[var(--color-line-soft)] pt-4">
                    <ShareButton machine={machine} models={ranked} />
                  </div>
                )}
              </>
            ) : (
              <p className="text-[13px] text-[var(--color-ink-faint)]">
                Ranking the catalog for this machine.
              </p>
            )}
          </div>
        )}
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card className="rising px-6 py-5" style={{ animationDelay: "90ms" }}>
          <Eyebrow>Measured</Eyebrow>
          <div className="mt-4 flex items-end justify-between gap-6">
            <div>
              <span className={measuring ? "" : "settling"} key={String(measured?.measured_at)}>
                <Figure
                  value={fmt.bandwidth(
                    machine.calibration.host.bandwidth_bytes_per_s,
                  )}
                  confidence={machine.calibration.source}
                  size="large"
                />
              </span>
              <p className="mt-1.5 text-[12px] text-[var(--color-ink-faint)]">
                {measuring
                  ? "Reading memory as fast as this machine allows…"
                  : measured
                    ? `Streaming bandwidth, measured ${fmt.since(measured.measured_at)}`
                    : "Assumed. Nothing has measured this machine."}
              </p>
            </div>
            {/* Quiet in both states. On a measured machine this is a
                secondary action; on an unmeasured one the invitation above
                is already the page's one primary call, and a second brass
                button beside it would make neither look like the thing to
                press. */}
            <MeasureButton
              measuring={measuring}
              onMeasure={onMeasure}
              label={measured ? "Measure again" : "Measure"}
              tone="quiet"
            />
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

        <Card className="rising px-6 py-5" style={{ animationDelay: "140ms" }}>
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
            {machine.pools.map((entry, index) => (
              <Spec
                key={entry.label}
                label={`${machine.display.pools[index] ?? entry.label}, usable`}
              >
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
        <Card className="rising px-6 py-5" style={{ animationDelay: "190ms" }}>
          <Eyebrow>How this machine was read</Eyebrow>
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

/**
 * The invitation, on a machine nothing has measured.
 *
 * Above the machine rather than beside it, because until this is done every
 * speed in the application is a guess, and a guess presented in the same type
 * as a measurement is the one thing this product must not do.
 */
function FirstRun({
  measuring,
  onMeasure,
}: {
  measuring: boolean;
  onMeasure: () => void;
}) {
  return (
    <Card
      className="rising flex flex-wrap items-center gap-x-8 gap-y-4 px-8 py-6"
      style={{
        animationDelay: "0ms",
        borderColor: "var(--color-brass-deep)",
        background:
          "linear-gradient(100deg, var(--color-brass-wash), var(--color-surface) 62%)",
      }}
    >
      <div className="min-w-[18rem] flex-1">
        <p className="font-display text-[19px] font-semibold leading-tight tracking-tight">
          Measure this machine first
        </p>
        <p className="mt-2 max-w-xl text-[13px] leading-relaxed text-[var(--color-ink-dim)]">
          Every speed below is an assumption until something times this
          hardware. It takes about a second, nothing leaves the machine, and
          afterwards each figure carries a solid rule instead of a dotted one.
        </p>
      </div>
      <MeasureButton
        measuring={measuring}
        onMeasure={onMeasure}
        label="Measure, one second"
        tone="primary"
      />
    </Card>
  );
}

function MeasureButton({
  measuring,
  onMeasure,
  label,
  tone,
}: {
  measuring: boolean;
  onMeasure: () => void;
  label: string;
  tone: "primary" | "quiet";
}) {
  const primary =
    "bg-[var(--color-brass)] text-[#17140d] hover:brightness-108 disabled:opacity-60";
  const quiet =
    "border border-[var(--color-line)] text-[var(--color-ink)] hover:bg-[var(--color-raised)] disabled:opacity-50";
  return (
    <button
      type="button"
      onClick={onMeasure}
      disabled={measuring}
      className={`shrink-0 rounded-lg px-4 py-2 text-[13px] font-medium transition-all active:scale-[0.98] ${
        tone === "primary" ? primary : quiet
      } ${measuring ? "sweeping" : ""}`}
    >
      {measuring ? "Measuring…" : label}
    </button>
  );
}

/** The engine's caveats, in words. */
function detectionNote(note: Record<string, unknown> & { note: string }): string {
  const s = (key: string) => String(note[key]);
  const n = (key: string) => Number(note[key]);
  switch (note.note) {
    case "ignored_virtual_adapter":
      return `${s("name")} was ignored: a display, not something that computes.`;
    case "integrated_graphics":
      return note.claimed_vram_bytes
        ? `${s("name")} shares the machine's memory rather than owning any. The ${fmt.bytes(
            n("claimed_vram_bytes"),
          )} of "video memory" the system reports is a window onto system RAM, not a separate pool, so the whole of it has been counted once.`
        : `${s("name")} shares the machine's memory; its pool is system RAM.`;
    case "legacy_integrated_graphics":
      return `${s(
        "name",
      )} is older than any usable compute back end, so the model will run on the processor. It reads the same memory either way, so nothing is lost by that.`;
    case "nvidia_unavailable":
      return "No NVIDIA graphics card here, so nothing was asked of the NVIDIA driver.";
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
