/**
 * The controls that change every answer.
 *
 * Kept in a bar of their own, above the view rather than inside it, because
 * context length and concurrency are not settings. They are half the
 * question. A model that fits at 8k and not at 128k has not changed; the ask
 * has.
 */
import type { Sizing, UseCase, Preference, RuntimeName } from "../engine";
import type { View } from "../App";
import * as fmt from "../format";
import { Segmented } from "../components/ui";

/** The context lengths people actually choose, as a slider's stops. */
const CONTEXTS = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144];

const USE_CASES: { value: UseCase; label: string }[] = [
  { value: "general", label: "General" },
  { value: "coding", label: "Coding" },
  { value: "reasoning", label: "Reasoning" },
  { value: "chat", label: "Chat" },
  { value: "agentic", label: "Agentic" },
  { value: "long_context", label: "Long context" },
];

const PREFERENCES: { value: Preference; label: string }[] = [
  { value: "quality", label: "Quality" },
  { value: "balanced", label: "Balanced" },
  { value: "speed", label: "Speed" },
];

/**
 * Every program the engine can size for and say how to start. LM Studio and
 * Ollama are llama.cpp underneath and are sized as it, but each wants the
 * file in a different place and starts it a different way, so they are
 * offered by name.
 */
const RUNTIMES: { value: RuntimeName; label: string }[] = [
  { value: "llama-cpp", label: "llama.cpp" },
  { value: "llama-cpp-no-flash", label: "llama.cpp, no flash attention" },
  { value: "lm-studio", label: "LM Studio" },
  { value: "ollama", label: "Ollama" },
  { value: "vllm", label: "vLLM" },
  { value: "mlx", label: "MLX" },
];

export function SizingBar({
  sizing,
  onChange,
  view,
}: {
  sizing: Sizing;
  onChange: (sizing: Sizing) => void;
  view: View;
}) {
  const set = <K extends keyof Sizing>(key: K, value: Sizing[K]) =>
    onChange({ ...sizing, [key]: value });

  const stop = Math.max(0, CONTEXTS.indexOf(sizing.context));

  return (
    <div className="flex shrink-0 flex-wrap items-center gap-x-7 gap-y-3 border-b border-[var(--color-line)] bg-[var(--color-ground)] px-7 py-3">
      <label className="flex items-center gap-3">
        <span className="text-[12px] text-[var(--color-ink-dim)]">Context</span>
        <input
          type="range"
          min={0}
          max={CONTEXTS.length - 1}
          step={1}
          value={stop}
          onChange={(event) =>
            set("context", CONTEXTS[Number(event.target.value)] ?? 8192)
          }
          className="w-36 accent-[var(--color-brass)]"
          aria-label="Context length"
        />
        <span className="figure w-11 text-[13px] font-medium">
          {fmt.tokens(sizing.context)}
        </span>
      </label>

      <label className="flex items-center gap-3">
        <span className="text-[12px] text-[var(--color-ink-dim)]">
          Concurrent
        </span>
        <input
          type="range"
          min={1}
          max={16}
          step={1}
          value={sizing.parallel}
          onChange={(event) => set("parallel", Number(event.target.value))}
          className="w-24 accent-[var(--color-brass)]"
          aria-label="Concurrent sequences"
        />
        <span className="figure w-5 text-[13px] font-medium">
          {sizing.parallel}
        </span>
      </label>

      {/* Only the views that rank or size a model are changed by these two. */}
      {view !== "Cost" && (
        <>
          <label className="flex items-center gap-2.5">
            <span className="text-[12px] text-[var(--color-ink-dim)]">For</span>
            <select
              value={sizing.use_case}
              onChange={(event) => set("use_case", event.target.value as UseCase)}
              className="rounded-lg border border-[var(--color-line)] bg-[var(--color-surface)] px-2 py-1 text-[12px]"
            >
              {USE_CASES.map((option) => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
          </label>

          <Segmented
            label="What to optimise for"
            value={sizing.preference}
            options={PREFERENCES}
            onChange={(value) => set("preference", value)}
          />
        </>
      )}

      <label className="ml-auto flex items-center gap-2.5">
        <span className="text-[12px] text-[var(--color-ink-dim)]">Runtime</span>
        <select
          value={sizing.runtime}
          onChange={(event) => set("runtime", event.target.value as RuntimeName)}
          className="rounded-lg border border-[var(--color-line)] bg-[var(--color-surface)] px-2 py-1 text-[12px]"
        >
          {RUNTIMES.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}
