/**
 * What to do once the sizing is settled.
 *
 * The Plan view used to end with a file. It ends with this: the command that
 * starts the model with the context, the layer split and the cache format
 * the solver chose, so the thing that runs is the thing that was sized. For
 * a runtime that does not run the file there is no download at all, and this
 * card takes the download's place and says why.
 *
 * Nothing here is composed. Every command and every line of the Modelfile is
 * a string the engine produced, quoted for this platform, and the notes are
 * put into words by the same table the command line uses. The window only
 * lays them out and offers to copy them.
 */
import { useState } from "react";
import * as engine from "../engine";
import type { Launch, Sizing } from "../engine";
import * as fmt from "../format";
import { Card, Eyebrow } from "./ui";
import { Minor } from "./Download";

export function LaunchCard({
  launch,
  id,
  sizing,
}: {
  launch: Launch;
  id: string;
  sizing: Sizing;
}) {
  const host = fmt.hostLabel(launch.host);
  const { weights } = launch;

  return (
    <Card className="px-6 py-5">
      <Eyebrow>
        {launch.runs_gguf ? `Then run it with ${host}` : `Run it with ${host}`}
      </Eyebrow>

      {weights.action === "host_fetches" && (
        <p className="mt-3 text-[13px] leading-relaxed text-[var(--color-ink-dim)]">
          Nothing to download from here. {host} fetches{" "}
          <span className="figure text-[var(--color-ink)]">{weights.repo}</span> as{" "}
          {weights.format} itself, from the model's own repository.
        </p>
      )}

      {weights.action === "search" && (
        <>
          <p className="mt-3 text-[13px] leading-relaxed text-[var(--color-ink-dim)]">
            Nothing to download from here. {host} runs {weights.format}, and the
            catalog does not record which conversion exists for this model, so
            search for one rather than trusting a guessed name:
          </p>
          <Line text={weights.url} />
          <p className="mt-3 text-[12px] leading-relaxed text-[var(--color-ink-faint)]">
            Then, with the repository you chose filled in:
          </p>
          <Line text={weights.command_shape} template />
        </>
      )}

      {weights.action === "no_build" && (
        <p className="mt-3 text-[13px] leading-relaxed text-[var(--color-ink-dim)]">
          No {weights.quant} build is published for this model, so there is
          nothing to fetch at the format that was sized. Any of the builds
          listed below can be fetched instead.
        </p>
      )}

      {launch.file && <ModelFile file={launch.file} id={id} sizing={sizing} />}

      {launch.commands.length > 0 && (
        <div className="mt-3 flex flex-col gap-2">
          {launch.commands.map((command) => (
            <Line key={command} text={command} />
          ))}
        </div>
      )}

      {launch.notes.length > 0 && (
        <ul className="mt-4 flex flex-col gap-2">
          {launch.notes.map((note, index) => (
            <li
              key={`${note.note}-${index}`}
              className="flex gap-3 text-[12px] leading-relaxed text-[var(--color-ink-faint)]"
            >
              <span
                className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-[var(--color-brass)]"
                aria-hidden
              />
              {fmt.launchNote(note, host)}
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

/** The file Ollama reads, shown as it will be written and written on request. */
function ModelFile({
  file,
  id,
  sizing,
}: {
  file: NonNullable<Launch["file"]>;
  id: string;
  sizing: Sizing;
}) {
  const [saved, setSaved] = useState<string | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  const save = async () => {
    setFailure(null);
    try {
      setSaved(await engine.saveLaunchFile(id, sizing));
    } catch (error) {
      setFailure(String(error));
    }
  };

  return (
    <div className="mt-3">
      <p className="text-[12px] text-[var(--color-ink-faint)]">
        Write <span className="figure text-[var(--color-ink-dim)]">{file.name}</span>{" "}
        beside the weights, saying:
      </p>
      <div className="mt-1.5 flex items-start gap-3 rounded-lg border border-[var(--color-line)] bg-[var(--color-ground)] px-3 py-2">
        <pre className="figure min-w-0 flex-1 overflow-x-auto whitespace-pre text-[12px] leading-relaxed">
          {file.content.trimEnd()}
        </pre>
        <span className="flex shrink-0 gap-2">
          <Copy text={file.content} />
          <Minor onClick={() => void save()}>Write it there</Minor>
        </span>
      </div>
      {saved && (
        <p className="mt-1.5 flex items-baseline gap-2 text-[11px] text-[var(--color-ink-faint)]">
          <span className="truncate">Written to {saved}</span>
          <button
            type="button"
            onClick={() => void engine.reveal(saved)}
            className="shrink-0 underline decoration-[var(--color-line)] underline-offset-2"
          >
            show in folder
          </button>
        </p>
      )}
      {failure && (
        <p className="mt-1.5 text-[11px]" style={{ color: "var(--color-over)" }}>
          {failure}
        </p>
      )}
    </div>
  );
}

/** One thing to type, with a way to take it. */
function Line({ text, template = false }: { text: string; template?: boolean }) {
  return (
    <div className="mt-1.5 flex items-center gap-3 rounded-lg border border-[var(--color-line)] bg-[var(--color-ground)] px-3 py-2">
      <code
        className={`figure min-w-0 flex-1 select-text overflow-x-auto whitespace-nowrap text-[12px] ${
          template ? "text-[var(--color-ink-faint)]" : ""
        }`}
      >
        {text}
      </code>
      {!template && <Copy text={text} />}
    </div>
  );
}

/**
 * Copy to the clipboard, and say whether it happened.
 *
 * The async clipboard API is solid in WebView2 and WKWebView and uneven in
 * WebKitGTK. When it refuses, the text is still on screen and selectable,
 * and the button says so instead of pretending.
 */
function Copy({ text }: { text: string }) {
  const [state, setState] = useState<"idle" | "copied" | "refused">("idle");

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setState("copied");
    } catch {
      setState("refused");
    }
    window.setTimeout(() => setState("idle"), 1800);
  };

  return (
    <Minor onClick={() => void copy()}>
      {state === "copied" ? "Copied" : state === "refused" ? "Select it" : "Copy"}
    </Minor>
  );
}
