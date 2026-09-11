/**
 * The engine, as the window sees it.
 *
 * Written against `crates/whatllm-app/src/api.rs` and checked by
 * `crates/whatllm-app/tests/contract.rs`, which asserts these field names
 * against real serialised output. Rename one in Rust without changing it here
 * and that test fails by name — because a hand-written type declaration is a
 * promise nothing else checks, and a broken one reads `undefined` and shows a
 * blank where a figure should be.
 *
 * Snake case throughout, matching the engine and `whatllm --json`. Nothing
 * here is computed: every number this window shows is one the engine decided.
 */
import { invoke } from "@tauri-apps/api/core";

/** How trustworthy a figure is, best to worst. */
export type Confidence =
  | "measured_here"
  | "calibrated"
  | "vendor_spec"
  | "fallback";

/** How comfortably a model sits in the pool it was placed in. */
export type Verdict = "comfortable" | "fits" | "tight" | "does_not_fit";

/** Where a model runs. Internally tagged; switch on `mode`. */
export type RunMode =
  | { mode: "accelerated"; devices: number }
  | { mode: "unified" }
  | { mode: "offloaded"; gpu_layers: number; total_layers: number }
  | { mode: "cpu" };

export type PoolKind = "device" | "host" | "unified";

export interface MemoryPool {
  label: string;
  kind: PoolKind;
  usable_bytes: number;
  devices: number[];
}

/** What a model needs, in the parts the engine accounts for separately. */
export interface Footprint {
  weights: number;
  /** True when a published file size backed the weight figure. */
  weights_measured: boolean;
  kv_cache: number;
  activations: number;
  /** Zero under flash attention, gigabytes without it. */
  attention_scores: number;
  runtime_overhead: number;
  headroom: number;
  /** Everything that must be resident, headroom excluded. */
  resident: number;
  /** What the pool must supply for this to load and stay up. */
  required: number;
}

/** Something worth saying about a placement. Tagged `note`; shapes vary. */
export type FitNote = { note: string } & Record<string, unknown>;

export interface FitView {
  quant: string;
  bits_per_weight: number;
  kv_quant: string;
  run_mode: RunMode;
  pool: MemoryPool;
  memory: Footprint;
  utilisation: number;
  verdict: Verdict;
  decode_tps: number;
  /** Null when nothing measured compute, which is not the same as zero. */
  prefill_tps: number | null;
  confidence: Confidence;
  quality: number;
  quality_full_precision: number;
  degradation: number;
  quality_coverage: number;
  max_context: number | null;
  score: number;
  notes: FitNote[];
  /**
   * How many catalog models this one scores at least as well as. Present only
   * from `rank`, where the whole field is in view; a bare score out of 100
   * means nothing without it, because nothing scores near 100 on the
   * benchmarks the scale averages.
   */
  quality_rank: { at_or_below: number; of: number } | null;
}

export interface Accelerator {
  index: number;
  name: string;
  vendor: string;
  backend: string;
  total_bytes: number;
  reserved_bytes: number;
  unified: boolean;
  drives_display: boolean;
  peak_bandwidth_gbps?: number;
  peak_tflops_fp16?: number;
}

export interface HostMemory {
  total_bytes: number;
  available_bytes: number;
  channels?: number;
  speed_mts?: number;
  /** Memory firmware kept from the OS for an integrated GPU, when measured. */
  uma_carveout_bytes?: number;
}

export interface SystemProfile {
  cpu: {
    brand: string;
    physical_cores: number;
    logical_cores: number;
    arch: string;
  };
  memory: HostMemory;
  accelerators: Accelerator[];
  os: string;
}

/** A caveat detection recorded. Tagged `note`; the rest varies. */
export type DetectionNote = { note: string } & Record<string, unknown>;

export interface Detection {
  system: SystemProfile;
  notes: DetectionNote[];
  raw_adapters: unknown[];
}

export interface Calibration {
  accelerator: { bandwidth_bytes_per_s: number; compute_flops: number | null } | null;
  host: { bandwidth_bytes_per_s: number; compute_flops: number | null };
  overhead_ms_per_token: number;
  source: Confidence;
}

export interface Machine {
  detection: Detection;
  calibration: Calibration;
  measurement: {
    fingerprint: string;
    host_bytes_per_s: number;
    single_thread_bytes_per_s: number;
    /** Seconds since the Unix epoch. */
    measured_at: number;
  } | null;
  pools: MemoryPool[];
  /** The same names with trademark marks removed, for showing to a person. */
  display: {
    cpu: string;
    accelerators: string[];
    /** In the same order as `pools`. */
    pools: string[];
  };
  catalog_source: string;
  catalog_size: number;
  catalog_generated: string;
}

export interface RankedModel {
  id: string;
  name: string;
  family: string;
  parameters: number;
  /** Read per token. Fewer than `parameters` for a sparse model. */
  active_parameters: number;
  sparse: boolean;
  max_trained_context: number;
  fit: FitView;
}

export interface Build {
  quant: string;
  bytes: number;
  command: string;
  chosen: boolean;
}

export interface CurveSample {
  context: number;
  required: number;
  decode_tps: number;
}

export interface Plan {
  id: string;
  name: string;
  family: string;
  license: string | null;
  released: string | null;
  fit: FitView;
  builds: Build[];
  curve: CurveSample[];
  pool_bytes: number;
}

/** Which side won, and by how much. Internally tagged; switch on `verdict`. */
export type CostVerdict =
  | { verdict: "local_cheaper"; by_percent: number }
  | { verdict: "api_cheaper"; by_percent: number }
  | { verdict: "too_close_to_call" };

export interface CostComparison {
  compute_hours_per_month: number;
  /** False when prompt processing is uncounted rather than free. */
  prefill_included: boolean;
  local_energy: number;
  local_amortisation: number;
  local_total: number;
  api_total: number;
  local_per_mtok: number;
  api_per_mtok: number;
  breakeven_requests_per_month: number | null;
  verdict: CostVerdict;
}

export interface Measurement {
  bytes_per_s: number;
  single_thread_bytes_per_s: number;
  threads: number;
  seconds: number;
  /** False when the machine measured its own contention. Not stored. */
  plausible: boolean;
  calibration: Calibration;
}

export type UseCase =
  | "general"
  | "coding"
  | "reasoning"
  | "math"
  | "chat"
  | "agentic"
  | "long_context"
  | "multilingual";

export type Preference = "quality" | "speed" | "balanced";

export type RuntimeName = "llama-cpp" | "llama-cpp-no-flash" | "vllm" | "mlx";

export interface Sizing {
  context: number;
  parallel: number;
  use_case: UseCase;
  preference: Preference;
  runtime: RuntimeName;
}

export interface CostQuery {
  requests: number;
  input: number;
  output: number;
  price_per_kwh: number;
  watts: number;
  hardware_cost: number;
  api_input: number;
  api_output: number;
}

/**
 * Where a transfer has got to. Internally tagged; switch on `state`.
 *
 * The one part of this application that reaches the network, and the only one
 * whose work outlives the view that started it — which is why it reports on an
 * event rather than by returning.
 */
export type Progress =
  | { state: "starting"; key: string }
  | {
      state: "running";
      key: string;
      received: number;
      total: number;
      bytes_per_s: number;
    }
  | { state: "paused"; key: string; received: number; total: number }
  | { state: "done"; key: string; path: string; bytes: number }
  | { state: "cancelled"; key: string }
  | { state: "failed"; key: string; reason: string };

export interface Destination {
  directory: string;
  path: string;
  /** Set when the finished file is already there at the recorded length. */
  present_bytes: number | null;
  /** Set when an interrupted transfer is waiting to be continued. */
  partial_bytes: number | null;
}

/** The event every running transfer reports itself on. */
export const DOWNLOAD_EVENT = "download:progress";

export const machine = () => invoke<Machine>("machine");
export const rank = (sizing: Sizing) => invoke<RankedModel[]>("rank", { sizing });
export const plan = (id: string, sizing: Sizing) =>
  invoke<Plan | null>("plan", { id, sizing });
export const cost = (id: string, sizing: Sizing, query: CostQuery) =>
  invoke<CostComparison | null>("cost", { id, sizing, query });
export const measure = (quick: boolean) =>
  invoke<Measurement>("measure", { quick });
export const destination = (id: string, quant: string) =>
  invoke<Destination>("destination", { id, quant });
export const downloadBuild = (id: string, quant: string) =>
  invoke<string>("download_build", { id, quant });
export const pauseDownload = (key: string) =>
  invoke<void>("pause_download", { key });
export const cancelDownload = (key: string) =>
  invoke<void>("cancel_download", { key });
export const downloads = () => invoke<Progress[]>("downloads");
export const reveal = (path: string) => invoke<void>("reveal", { path });
export const saveImage = (data: string, name: string) =>
  invoke<string>("save_image", { data, name });
