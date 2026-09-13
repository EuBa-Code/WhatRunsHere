/**
 * The window.
 *
 * Four views over one machine and one catalog. The sizing controls (context,
 * concurrency, what the model is for) belong to the whole application rather
 * than to any view, because changing one changes every answer, and a control
 * that lived inside a tab would make it look as though it did not.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import * as engine from "./engine";
import type { Installed, Machine, RankedModel, Sizing } from "./engine";
import * as fmt from "./format";
import { ProvenanceLegend } from "./components/ui";
import { TitleBar } from "./components/TitleBar";
import { MachineView } from "./views/MachineView";
import { ModelsView } from "./views/ModelsView";
import { PlanView } from "./views/PlanView";
import { CostView } from "./views/CostView";
import { SizingBar } from "./views/SizingBar";

const VIEWS = ["Machine", "Models", "Plan", "Cost"] as const;
export type View = (typeof VIEWS)[number];

const DEFAULT_SIZING: Sizing = {
  context: 8192,
  parallel: 1,
  use_case: "general",
  preference: "balanced",
  runtime: "llama-cpp",
};

/** Where a viewer's own preferences are kept between sessions. */
const THEME_KEY = "whatrunshere.theme";

export default function App() {
  const [view, setView] = useState<View>("Machine");
  const [sizing, setSizing] = useState<Sizing>(DEFAULT_SIZING);
  const [machine, setMachine] = useState<Machine | null>(null);
  const [ranked, setRanked] = useState<RankedModel[] | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [measuring, setMeasuring] = useState(false);
  const [failure, setFailure] = useState<string | null>(null);
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    try {
      return localStorage.getItem(THEME_KEY) === "light" ? "light" : "dark";
    } catch {
      return "dark";
    }
  });

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      // A viewer with site data blocked keeps the default. Nothing else breaks.
    }
  }, [theme]);

  const loadMachine = useCallback(async () => {
    try {
      setMachine(await engine.machine());
    } catch (error) {
      setFailure(String(error));
    }
  }, []);

  useEffect(() => {
    void loadMachine();
  }, [loadMachine]);

  // Re-ranked whenever the sizing changes, because it is the sizing that
  // decides the order. 56 models solve in a few milliseconds.
  useEffect(() => {
    let current = true;
    engine
      .rank(sizing)
      .then((result) => {
        if (current) setRanked(result);
      })
      .catch((error) => current && setFailure(String(error)));
    return () => {
      current = false;
    };
  }, [sizing]);

  // What is already on the disk, and how each of it runs at this sizing. The
  // disk is read once; the fits follow the sizing like the ranking does.
  const [installed, setInstalled] = useState<Installed | null>(null);
  useEffect(() => {
    let current = true;
    engine
      .installed(sizing, false)
      .then((result) => {
        if (current) setInstalled(result);
      })
      .catch(() => {
        // A disk that cannot be read is an absence, not a failure of the
        // window: the section is simply not shown.
      });
    return () => {
      current = false;
    };
  }, [sizing]);
  const refreshInstalled = useCallback(() => {
    engine
      .installed(sizing, true)
      .then(setInstalled)
      .catch(() => {});
  }, [sizing]);

  // The selection follows the ranking until someone picks for themselves.
  const selectedId = useMemo(
    () => selected ?? ranked?.[0]?.id ?? null,
    [selected, ranked],
  );

  const runProbe = useCallback(async () => {
    setMeasuring(true);
    try {
      await engine.measure(false);
      await loadMachine();
      // Every estimate rests on the calibration, so the ranking is stale the
      // moment it changes.
      setRanked(await engine.rank(sizing));
    } catch (error) {
      setFailure(String(error));
    } finally {
      setMeasuring(false);
    }
  }, [loadMachine, sizing]);

  const open = useCallback((id: string, next: View = "Plan") => {
    setSelected(id);
    setView(next);
  }, []);

  if (failure) {
    return (
      <main className="grid h-full place-items-center p-10">
        <div className="card max-w-lg px-8 py-7">
          <p className="font-display text-[17px] font-semibold">
            The engine did not answer
          </p>
          <p className="mt-2 text-[13px] leading-relaxed text-[var(--color-ink-dim)]">
            {failure}
          </p>
          <button
            type="button"
            onClick={() => {
              setFailure(null);
              void loadMachine();
            }}
            className="mt-5 rounded-lg border border-[var(--color-line)] px-3 py-1.5 text-[13px] hover:bg-[var(--color-raised)]"
          >
            Try again
          </button>
        </div>
      </main>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <TopBar
        view={view}
        onView={setView}
        machine={machine}
        theme={theme}
        onTheme={() => setTheme(theme === "dark" ? "light" : "dark")}
      />

      {view !== "Machine" && (
        <SizingBar sizing={sizing} onChange={setSizing} view={view} />
      )}

      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-[1120px] px-7 py-7">
          {view === "Machine" && (
            <MachineView
              machine={machine}
              measuring={measuring}
              onMeasure={runProbe}
              best={ranked?.[0] ?? null}
              ranked={ranked}
              onOpen={open}
            />
          )}
          {view === "Models" && (
            <ModelsView
              ranked={ranked}
              sizing={sizing}
              selectedId={selectedId}
              installed={installed}
              onOpen={open}
            />
          )}
          {view === "Plan" && (
            <PlanView
              id={selectedId}
              sizing={sizing}
              ranked={ranked}
              installed={installed}
              onInstalledChanged={refreshInstalled}
              onOpen={open}
            />
          )}
          {view === "Cost" && (
            <CostView id={selectedId} sizing={sizing} ranked={ranked} onOpen={open} />
          )}
        </div>
      </main>
    </div>
  );
}

function TopBar({
  view,
  onView,
  machine,
  theme,
  onTheme,
}: {
  view: View;
  onView: (view: View) => void;
  machine: Machine | null;
  theme: "dark" | "light";
  onTheme: () => void;
}) {
  const measured = machine?.measurement;
  return (
    <TitleBar>
      <div className="flex items-center gap-2.5">
        <Mark />
        <span className="font-display text-[14px] font-semibold tracking-tight">
          WhatRunsHere
        </span>
      </div>

      <nav className="flex items-center gap-0.5">
        {VIEWS.map((name) => (
          <button
            key={name}
            type="button"
            onClick={() => onView(name)}
            aria-current={view === name ? "page" : undefined}
            className={`rounded-lg px-2.5 py-1 text-[13px] transition-colors ${
              view === name
                ? "bg-[var(--color-raised)] text-[var(--color-ink)]"
                : "text-[var(--color-ink-faint)] hover:text-[var(--color-ink-dim)]"
            }`}
          >
            {name}
          </button>
        ))}
      </nav>

      <div className="ml-auto flex items-center gap-5">
        <ProvenanceLegend />
        {machine && (
          <span
            className="text-[11px] text-[var(--color-ink-faint)]"
            title={
              measured
                ? `Bandwidth measured ${fmt.since(measured.measured_at)}`
                : "Nothing has measured this machine yet"
            }
          >
            <span
              className={`figure ${
                measured ? "prov prov-measured" : "prov prov-assumed"
              }`}
            >
              {fmt.bandwidth(machine.calibration.host.bandwidth_bytes_per_s)}
            </span>
          </span>
        )}
        <button
          type="button"
          onClick={onTheme}
          aria-label={theme === "dark" ? "Use the light theme" : "Use the dark theme"}
          className="rounded-lg px-1.5 py-1 text-[13px] text-[var(--color-ink-faint)] transition-colors hover:text-[var(--color-ink)]"
        >
          {theme === "dark" ? "◐" : "◑"}
        </button>
      </div>
    </TitleBar>
  );
}

/** The application mark: the memory column, small enough to sit in a bar. */
function Mark() {
  return (
    <svg width="16" height="18" viewBox="0 0 16 18" aria-hidden>
      <rect
        x="0.75"
        y="0.75"
        width="14.5"
        height="16.5"
        rx="3"
        fill="none"
        stroke="var(--color-line)"
        strokeWidth="1.5"
      />
      <rect x="3" y="10" width="10" height="5" rx="1" fill="var(--color-band-weights)" />
      <rect x="3" y="7" width="10" height="2.2" rx="1" fill="var(--color-band-cache)" />
      <rect
        x="3"
        y="5.2"
        width="10"
        height="1.2"
        rx="0.6"
        fill="var(--color-band-activations)"
      />
    </svg>
  );
}
