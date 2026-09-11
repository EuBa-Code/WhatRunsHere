/**
 * Turning the engine's numbers into words.
 *
 * The one rule here: never round away a distinction someone would act on. A
 * model at 7.9 GiB and one at 8.1 GiB are different answers on an 8 GiB card,
 * so sizes near a boundary keep a decimal that a tidier format would drop.
 */
import type { Confidence, Host, LaunchNote, Verdict } from "./engine";

const KIB = 1024;

/** Bytes in the binary units every runtime and driver reports. */
export function bytes(value: number): string {
  if (!Number.isFinite(value) || value < 0) return "n/a";
  if (value < KIB) return `${Math.round(value)} B`;
  const units = ["KiB", "MiB", "GiB", "TiB"];
  let scaled = value / KIB;
  let unit = 0;
  while (scaled >= KIB && unit < units.length - 1) {
    scaled /= KIB;
    unit += 1;
  }
  // One decimal below a hundred, none above: 8.4 GiB is a decision, 847 GiB
  // is a fact, and 847.3 GiB is noise.
  const digits = scaled < 100 ? 1 : 0;
  return `${scaled.toFixed(digits)} ${units[unit]}`;
}

/** Bytes per second as the figure people quote. */
export function bandwidth(bytesPerSecond: number): string {
  if (!Number.isFinite(bytesPerSecond) || bytesPerSecond <= 0) return "n/a";
  return `${(bytesPerSecond / 1e9).toFixed(1)} GB/s`;
}

/** A token count, in the units contexts are named in. */
export function tokens(count: number): string {
  if (count >= 1024 * 1024) return `${(count / (1024 * 1024)).toFixed(1)}M`;
  if (count >= 1024) {
    const k = count / 1024;
    return `${Number.isInteger(k) ? k : k.toFixed(1)}k`;
  }
  return String(count);
}

/** A parameter count, as the model's name would give it. */
export function params(count: number): string {
  if (count >= 1e12) return `${(count / 1e12).toFixed(1)}T`;
  if (count >= 1e9) {
    const b = count / 1e9;
    return `${b >= 10 ? Math.round(b) : b.toFixed(1)}B`;
  }
  return `${Math.round(count / 1e6)}M`;
}

/** Generation speed. Below ten the tenths matter; above it they do not. */
export function tps(value: number): string {
  if (!Number.isFinite(value)) return "n/a";
  return value < 10 ? value.toFixed(1) : String(Math.round(value));
}

export function percent(fraction: number, digits = 0): string {
  if (!Number.isFinite(fraction)) return "n/a";
  return `${(fraction * 100).toFixed(digits)}%`;
}

/** Money, to the cent, in whatever currency the prices were given in. */
export function money(value: number): string {
  if (!Number.isFinite(value)) return "n/a";
  return value.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

/** A transfer rate, in the units a connection is sold in. */
export function rate(bytesPerSecond: number): string {
  if (!Number.isFinite(bytesPerSecond) || bytesPerSecond <= 0) return "n/a";
  const mb = bytesPerSecond / (1024 * 1024);
  return mb >= 10 ? `${Math.round(mb)} MB/s` : `${mb.toFixed(1)} MB/s`;
}

/**
 * How much longer, rounded honestly.
 *
 * Never to the second: a rate that swings with the connection makes a figure
 * that precise a lie that updates five times a second. Minutes, and "under a
 * minute" rather than counting down to zero.
 */
export function remaining(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "";
  if (seconds < 60) return "under a minute left";
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `about ${minutes} min left`;
  const hours = seconds / 3600;
  return hours < 2 ? "over an hour left" : `about ${Math.round(hours)} hours left`;
}

/** How long ago something happened, in the words a person would use. */
export function since(secondsSinceEpoch: number): string {
  const elapsed = Math.max(0, Date.now() / 1000 - secondsSinceEpoch);
  if (elapsed < 90) return "just now";
  if (elapsed < 3600) return `${Math.round(elapsed / 60)} minutes ago`;
  if (elapsed < 86400) {
    const hours = Math.round(elapsed / 3600);
    return hours === 1 ? "an hour ago" : `${hours} hours ago`;
  }
  const days = Math.round(elapsed / 86400);
  return days === 1 ? "yesterday" : `${days} days ago`;
}

/**
 * What a confidence tier means, and how the provenance rule under a figure
 * should be drawn.
 *
 * The three rules are the window's whole claim about honesty, so the mapping
 * lives in one place rather than at each call site.
 */
export function provenance(confidence: Confidence): {
  className: string;
  label: string;
  detail: string;
} {
  switch (confidence) {
    case "measured_here":
      return {
        className: "prov prov-measured",
        label: "Measured",
        detail: "Timed on this machine, running this model.",
      };
    case "calibrated":
      return {
        className: "prov prov-measured",
        label: "Calibrated",
        detail:
          "Built on bandwidth measured on this machine, by a probe that ran here.",
      };
    case "vendor_spec":
      return {
        className: "prov prov-spec",
        label: "From specification",
        detail:
          "Built on the bandwidth this device reports it can reach, not on one anything measured.",
      };
    case "fallback":
      return {
        className: "prov prov-assumed",
        label: "Assumed",
        detail:
          "Nothing measured this machine and nothing reported it. Measure it to replace this.",
      };
  }
}

/** How a verdict should read, and which signal colour carries it. */
export function verdict(value: Verdict): { label: string; tone: string } {
  switch (value) {
    case "comfortable":
      return { label: "Comfortable", tone: "good" };
    case "fits":
      return { label: "Fits", tone: "good" };
    case "tight":
      return { label: "Tight", tone: "tight" };
    case "does_not_fit":
      return { label: "Does not fit", tone: "over" };
  }
}

/** The engine's own phrasing for the notes it attaches to a placement. */
export function fitNote(note: Record<string, unknown> & { note: string }): string {
  const n = (key: string) => Number(note[key]);
  switch (note.note) {
    case "quantise_cache":
      return `A ${String(note.to).toUpperCase()} cache would free ${bytes(
        n("saves_bytes"),
      )}, at almost no cost to quality.`;
    case "context_ceiling":
      return `${tokens(n("requested"))} tokens does not fit. ${tokens(
        n("achievable"),
      )} does.`;
    case "enable_flash_attention":
      return `Flash attention would save ${bytes(
        n("saves_bytes"),
      )}: the attention score matrix, which is quadratic in context.`;
    case "quantise_to_avoid_offload":
      return `${String(note.to)} would keep every layer on the accelerator. Spilling layers to system memory costs far more than the format does.`;
    case "room_for_better_quant":
      return `${String(note.to)} also fits, and is worth ${n("quality_gain").toFixed(
        1,
      )} points of quality.`;
    case "running_close":
      return `${percent(n("utilisation"))} of the pool. It will load, but little else can.`;
    default:
      return note.note.replace(/_/g, " ");
  }
}

/** The name a person knows a host by. */
export function hostLabel(host: Host): string {
  switch (host) {
    case "llama_cpp":
      return "llama.cpp";
    case "lm_studio":
      return "LM Studio";
    case "ollama":
      return "Ollama";
    case "vllm":
      return "vLLM";
    case "mlx":
      return "MLX";
  }
}

/**
 * The engine's own phrasing for what it says about starting a model. The
 * same facts the command line prints, in the same words.
 */
export function launchNote(note: LaunchNote, host: string): string {
  const n = (key: string) => Number(note[key]);
  const s = (key: string) => String(note[key]);
  switch (note.note) {
    case "not_this_format":
      return `GGUF is not ${host}'s format (${s(
        "host_format",
      )}). The sizing above describes the same model in a different container.`;
    case "full_precision_weights":
      return `The original weights are 16-bit: about ${bytes(
        n("bytes"),
      )}, not the build sized above.`;
    case "host_needs":
      return `${host} needs ${s("platform")}, which this machine is not.`;
    case "host_not_installed":
      return `${host} does not seem to be installed: ${s(
        "looked_in",
      )} is not there. The file goes to the usual folder instead.`;
    case "appears_in_host":
      return `Once it has landed it appears in ${host} under ${s("name")}.`;
    case "second_copy":
      return `${host} copies the file into its own store, so the disk ends up holding two: ${bytes(
        n("bytes"),
      )} more.`;
    case "host_environment": {
      const variables = (note.variables as { name: string; value: string }[])
        .map((v) => `${v.name}=${v.value}`)
        .join(" ");
      return `Before starting ${host}, set ${variables}. The sizing assumed them.`;
    }
    case "cache_format_unavailable":
      return `The cache was sized as ${s(
        "sized",
      )}, which ${host} does not offer. It has ${(note.offers as string[]).join(
        ", ",
      )}; a wider one needs more memory than sized, a narrower one less.`;
    case "v_cache_uncompressed":
      return `Without flash attention the value half of the cache stays at 16 bits: about ${bytes(
        n("extra_bytes"),
      )} more than sized.`;
    case "set_context_in_host":
      return `${host} sets the context length in its own window. Set it to ${tokens(
        n("tokens"),
      )}, which is what was sized; its default is not.`;
    default:
      return note.note.replace(/_/g, " ");
  }
}
