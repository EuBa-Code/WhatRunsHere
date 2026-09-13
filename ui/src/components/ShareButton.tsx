/**
 * One button, three outcomes, and it says which one happened.
 *
 * Copying an image to the clipboard is not reliably available everywhere: the
 * async clipboard API is solid in WebView2 and WKWebView and uneven in
 * WebKitGTK, and a permission can be refused on any of them. So the button
 * tries the clipboard, falls back to writing a file, and tells you which it
 * did rather than flashing the same tick either way.
 */
import { useState } from "react";
import type { Machine, RankedModel } from "../engine";
import * as engine from "../engine";
import { render } from "./ShareCard";

type State =
  | { at: "idle" }
  | { at: "working" }
  | { at: "copied" }
  | { at: "saved"; path: string }
  | { at: "failed"; why: string };

export function ShareButton({
  machine,
  models,
}: {
  machine: Machine;
  models: RankedModel[];
}) {
  const [state, setState] = useState<State>({ at: "idle" });

  const share = async () => {
    setState({ at: "working" });
    try {
      const png = await render(machine, models);

      // The clipboard first, because the whole point is pasting it into a
      // conversation that is already open.
      try {
        await navigator.clipboard.write([
          new ClipboardItem({ "image/png": png }),
        ]);
        setState({ at: "copied" });
        window.setTimeout(() => setState({ at: "idle" }), 2600);
        return;
      } catch {
        // Falls through to the file. Not reported: the clipboard being
        // unavailable is not a failure anybody needs to read about when the
        // other path works.
      }

      const base64 = await toBase64(png);
      const path = await engine.saveImage(
        base64,
        `whatrunshere-${models[0]?.name ?? "result"}`,
      );
      setState({ at: "saved", path });
    } catch (error) {
      setState({ at: "failed", why: String(error) });
    }
  };

  const label = {
    idle: "Copy result as an image",
    working: "Drawing…",
    copied: "Copied. Paste it anywhere.",
    saved: "Saved as an image",
    failed: "Try again",
  }[state.at];

  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
      <button
        type="button"
        onClick={() => void share()}
        disabled={state.at === "working"}
        className={`inline-flex shrink-0 items-center gap-2 rounded-lg border px-3 py-1.5 text-[13px] transition-colors disabled:opacity-60 ${
          state.at === "copied"
            ? "border-[var(--color-good)] text-[var(--color-good)]"
            : "border-[var(--color-line)] hover:bg-[var(--color-raised)]"
        }`}
      >
        {state.at === "copied" ? <Tick /> : <Picture />}
        {label}
      </button>

      {state.at === "saved" && (
        <button
          type="button"
          onClick={() => void engine.reveal(state.path)}
          className="text-[12px] text-[var(--color-ink-faint)] underline decoration-[var(--color-line)] underline-offset-2 transition-colors hover:text-[var(--color-ink)]"
        >
          Show it in the folder
        </button>
      )}
      {state.at === "failed" && (
        <span className="text-[12px] text-[var(--color-over)]">{state.why}</span>
      )}
    </div>
  );
}

async function toBase64(blob: Blob): Promise<string> {
  const buffer = new Uint8Array(await blob.arrayBuffer());
  let binary = "";
  // In chunks: spreading a few hundred thousand bytes into one call overflows
  // the argument limit on every engine that has one.
  const CHUNK = 0x8000;
  for (let i = 0; i < buffer.length; i += CHUNK) {
    binary += String.fromCharCode(...buffer.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

function Picture() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
      <rect
        x="0.75"
        y="1.75"
        width="12.5"
        height="10.5"
        rx="2"
        fill="none"
        stroke="currentColor"
      />
      <circle cx="4.6" cy="5.4" r="1.2" fill="currentColor" />
      <path
        d="M1.4 11.2L5 7.6L7.4 10L9.6 8.2L12.6 11.2"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.2"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function Tick() {
  return (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden>
      <path
        d="M2.5 7.4L5.6 10.4L11.5 3.8"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
