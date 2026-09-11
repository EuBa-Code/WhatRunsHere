/**
 * Getting the weights.
 *
 * The one control in this application that reaches the network, and the only
 * one whose work outlives the view it was started from: a transfer keeps going
 * while you move between Models, Plan and Cost, and reports itself on an event
 * rather than through whatever component happens to be mounted.
 *
 * Which is why the state lives in a hook shared by everything that shows it,
 * seeded from the Rust side on mount. Events are fire-and-forget, so a window
 * that was on another view when a transfer finished would otherwise still be
 * drawing a bar that had stopped moving.
 */
import { useCallback, useEffect, useState } from "react";
import { listen } from "@tauri-apps/api/event";
import * as engine from "../engine";
import type { Build, Progress } from "../engine";
import * as fmt from "../format";

/** Every transfer this session has started, by build key. */
export function useDownloads() {
  const [byKey, setByKey] = useState<Record<string, Progress>>({});

  useEffect(() => {
    let live = true;
    // Seeded first, so a transfer that finished while this view was unmounted
    // is shown as finished rather than as never started.
    void engine.downloads().then((all) => {
      if (!live) return;
      setByKey(Object.fromEntries(all.map((p) => [p.key, p])));
    });

    const stop = listen<Progress>(engine.DOWNLOAD_EVENT, (event) => {
      setByKey((current) => ({ ...current, [event.payload.key]: event.payload }));
    });
    return () => {
      live = false;
      void stop.then((off) => off());
    };
  }, []);

  return byKey;
}

export const buildKey = (id: string, quant: string) => `${id}@${quant}`;

/**
 * The download control for one build.
 *
 * Five states, and each says what it is rather than showing a spinner: not
 * started, running, paused, already on disk, or failed with the reason.
 */
export function Download({
  id,
  quant,
  bytes,
  progress,
  builds,
}: {
  id: string;
  quant: string;
  bytes: number;
  progress: Progress | undefined;
  /** Every published build, so a smaller one can be offered when it fails. */
  builds: Build[];
}) {
  const key = buildKey(id, quant);
  const [where, setWhere] = useState<engine.Destination | null>(null);
  const [busy, setBusy] = useState(false);

  const look = useCallback(() => {
    void engine
      .destination(id, quant)
      .then(setWhere)
      .catch(() => setWhere(null));
  }, [id, quant]);

  // Re-checked whenever a transfer settles, because what is on disk changed.
  useEffect(look, [look]);
  useEffect(() => {
    if (progress?.state === "done" || progress?.state === "cancelled") look();
  }, [progress?.state, look]);

  const start = async () => {
    setBusy(true);
    try {
      await engine.downloadBuild(id, quant);
    } finally {
      setBusy(false);
    }
  };

  // Already on disk and the right length: there is nothing to fetch.
  if (where?.present_bytes && progress?.state !== "running") {
    return (
      <Shell>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <span className="flex items-center gap-2 text-[13px]">
            <Tick />
            <span className="font-medium">Downloaded</span>
            <span className="figure text-[var(--color-ink-dim)]">
              {fmt.bytes(where.present_bytes)}
            </span>
          </span>
          <button
            type="button"
            onClick={() => void engine.reveal(where.path)}
            className="ml-auto rounded-lg border border-[var(--color-line)] px-3 py-1.5 text-[13px] transition-colors hover:bg-[var(--color-raised)]"
          >
            Show in folder
          </button>
        </div>
        <Path where={where} />
      </Shell>
    );
  }

  switch (progress?.state) {
    case "starting":
      return (
        <Shell>
          <p className="text-[13px] text-[var(--color-ink-dim)]">
            Asking huggingface.co for the file…
          </p>
        </Shell>
      );

    case "running":
      return (
        <Shell>
          <Bar received={progress.received} total={progress.total} />
          <div className="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-2">
            <span className="figure text-[12px] text-[var(--color-ink-dim)]">
              {fmt.bytes(progress.received)} of {fmt.bytes(progress.total)}
              {progress.bytes_per_s > 0 && (
                <>
                  {" · "}
                  {fmt.rate(progress.bytes_per_s)}
                  {" · "}
                  {fmt.remaining(
                    (progress.total - progress.received) / progress.bytes_per_s,
                  )}
                </>
              )}
            </span>
            <span className="ml-auto flex gap-2">
              <Minor onClick={() => void engine.pauseDownload(key)}>Pause</Minor>
              <Minor onClick={() => void engine.cancelDownload(key)}>Cancel</Minor>
            </span>
          </div>
          <Path where={where} />
        </Shell>
      );

    case "paused":
      return (
        <Shell>
          <Bar received={progress.received} total={progress.total} muted />
          <div className="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-2">
            <span className="figure text-[12px] text-[var(--color-ink-dim)]">
              Paused at {fmt.bytes(progress.received)} of {fmt.bytes(progress.total)}
            </span>
            <span className="ml-auto flex gap-2">
              <Primary onClick={() => void start()} busy={busy}>
                Continue
              </Primary>
              <Minor onClick={() => void engine.cancelDownload(key)}>Discard</Minor>
            </span>
          </div>
        </Shell>
      );

    case "failed":
      return (
        <Shell tone="over">
          <p className="text-[13px] font-medium">The download did not finish</p>
          <p className="mt-1.5 text-[12px] leading-relaxed text-[var(--color-ink-dim)]">
            {progress.reason}
          </p>
          <div className="mt-3 flex gap-2">
            <Primary onClick={() => void start()} busy={busy}>
              Try again
            </Primary>
          </div>
        </Shell>
      );

    default:
      return (
        <Shell>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-3">
            <div className="min-w-0">
              <p className="text-[13px]">
                <span className="font-medium">Download {quant}</span>{" "}
                <span className="figure text-[var(--color-ink-dim)]">
                  {fmt.bytes(bytes)}
                </span>
                {builds.length > 1 && (
                  <span className="text-[var(--color-ink-faint)]">
                    {" · "}
                    {builds.length} formats published
                  </span>
                )}
              </p>
              {where?.partial_bytes ? (
                <p className="mt-1 text-[12px] text-[var(--color-ink-faint)]">
                  {fmt.bytes(where.partial_bytes)} of a previous attempt is still
                  here and will be continued rather than fetched again.
                </p>
              ) : null}
            </div>
            <span className="ml-auto">
              <Primary onClick={() => void start()} busy={busy}>
                {where?.partial_bytes ? "Continue download" : "Download"}
              </Primary>
            </span>
          </div>
          <Path where={where} />
        </Shell>
      );
  }
}

function Shell({
  children,
  tone,
}: {
  children: React.ReactNode;
  tone?: "over";
}) {
  return (
    <div
      className="card px-5 py-4"
      style={
        tone === "over"
          ? { borderColor: "var(--color-over)" }
          : { borderColor: "var(--color-line)" }
      }
    >
      {children}
    </div>
  );
}

function Bar({
  received,
  total,
  muted = false,
}: {
  received: number;
  total: number;
  muted?: boolean;
}) {
  const fraction = total > 0 ? Math.min(1, received / total) : 0;
  return (
    <div
      className="h-1.5 w-full overflow-hidden rounded-full"
      style={{ background: "var(--color-line-soft)" }}
      role="progressbar"
      aria-valuenow={Math.round(fraction * 100)}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div
        className="h-full rounded-full transition-[width] duration-200 ease-out"
        style={{
          width: `${fraction * 100}%`,
          background: muted ? "var(--color-ink-faint)" : "var(--color-brass)",
        }}
      />
    </div>
  );
}

/** Where it is going, stated before it goes rather than after. */
function Path({ where }: { where: engine.Destination | null }) {
  if (!where) return null;
  return (
    <p className="mt-2.5 truncate text-[11px] text-[var(--color-ink-faint)]">
      {where.path}
    </p>
  );
}

function Primary({
  children,
  onClick,
  busy,
}: {
  children: React.ReactNode;
  onClick: () => void;
  busy: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={busy}
      className="shrink-0 rounded-lg bg-[var(--color-brass)] px-4 py-2 text-[13px] font-medium text-[#17140d] transition-all hover:brightness-108 active:scale-[0.98] disabled:opacity-60"
    >
      {children}
    </button>
  );
}

function Minor({
  children,
  onClick,
}: {
  children: React.ReactNode;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="shrink-0 rounded-lg border border-[var(--color-line)] px-3 py-1.5 text-[12px] transition-colors hover:bg-[var(--color-raised)]"
    >
      {children}
    </button>
  );
}

function Tick() {
  return (
    <svg width="13" height="13" viewBox="0 0 13 13" aria-hidden>
      <circle cx="6.5" cy="6.5" r="6" fill="none" stroke="var(--color-good)" />
      <path
        d="M3.5 6.8L5.6 8.9L9.5 4.6"
        fill="none"
        stroke="var(--color-good)"
        strokeWidth="1.4"
        strokeLinecap="round"
      />
    </svg>
  );
}
