/**
 * Where the weights go.
 *
 * This is the setting that matters most on a machine with more than one disk,
 * because a model is tens of gigabytes and the system drive is frequently the
 * small fast one. So the disks are listed with their free space rather than
 * hidden behind a folder picker: picking a disk is the decision, and picking a
 * folder is the exception.
 *
 * Free space is shown against the size of the build being fetched, so a disk
 * that cannot hold it says so before anything starts rather than after twenty
 * gigabytes have arrived.
 */
import { useCallback, useEffect, useState } from "react";
import { open } from "@tauri-apps/plugin-dialog";
import * as engine from "../engine";
import type { Volume } from "../engine";
import * as fmt from "../format";

export function Destination({
  directory,
  freeBytes,
  needsBytes,
  onChanged,
  fixedBy,
}: {
  directory: string;
  freeBytes: number | null;
  /** The build about to be fetched, so the room left can be judged. */
  needsBytes: number;
  onChanged: () => void;
  /**
   * Who decided the directory, when it was not the user: LM Studio reads
   * only its own folder, so the file goes there and the choice is not
   * offered. Shown as a fact rather than hidden as a missing control.
   */
  fixedBy?: string;
}) {
  const [open_, setOpen] = useState(false);
  const [volumes, setVolumes] = useState<Volume[] | null>(null);
  const [failure, setFailure] = useState<string | null>(null);

  const look = useCallback(() => {
    void engine
      .volumes()
      .then(setVolumes)
      .catch(() => setVolumes([]));
  }, []);

  useEffect(() => {
    if (open_ && volumes === null) look();
  }, [open_, volumes, look]);

  const choose = async (path: string | null) => {
    setFailure(null);
    try {
      await engine.setDownloadDir(path);
      setVolumes(null);
      setOpen(false);
      onChanged();
    } catch (error) {
      setFailure(String(error));
    }
  };

  const browse = async () => {
    const picked = await open({
      directory: true,
      multiple: false,
      title: "Where should models be saved?",
    });
    if (typeof picked === "string") await choose(picked);
  };

  const tight = freeBytes !== null && freeBytes < needsBytes;

  if (fixedBy) {
    return (
      <div className="mt-2.5">
        <p className="flex items-baseline gap-2 text-[11px] text-[var(--color-ink-faint)]">
          <span className="truncate">{directory}</span>
          {freeBytes !== null && (
            <span
              className="figure shrink-0"
              style={tight ? { color: "var(--color-over)" } : undefined}
            >
              {fmt.bytes(freeBytes)} free
            </span>
          )}
          <span className="shrink-0">{fixedBy}</span>
        </p>
        {tight && (
          <p className="mt-1 text-[11px]" style={{ color: "var(--color-over)" }}>
            This disk has less room than the file needs.
          </p>
        )}
      </div>
    );
  }

  return (
    <div className="mt-2.5">
      <button
        type="button"
        onClick={() => setOpen(!open_)}
        className="flex w-full items-baseline gap-2 text-left text-[11px] text-[var(--color-ink-faint)] transition-colors hover:text-[var(--color-ink-dim)]"
      >
        <span className="truncate">{directory}</span>
        {freeBytes !== null && (
          <span
            className="figure shrink-0"
            style={tight ? { color: "var(--color-over)" } : undefined}
          >
            {fmt.bytes(freeBytes)} free
          </span>
        )}
        <span className="shrink-0 underline decoration-[var(--color-line)] underline-offset-2">
          {open_ ? "close" : "change"}
        </span>
      </button>

      {tight && !open_ && (
        <p className="mt-1 text-[11px]" style={{ color: "var(--color-over)" }}>
          This disk has less room than the file needs. Choose another below.
        </p>
      )}

      {open_ && (
        <div className="mt-2.5 rounded-lg border border-[var(--color-line)] bg-[var(--color-ground)] p-1">
          {volumes === null && (
            <p className="px-2.5 py-2 text-[12px] text-[var(--color-ink-faint)]">
              Reading the disks…
            </p>
          )}

          {volumes?.map((volume) => {
            const fits = volume.free_bytes >= needsBytes;
            return (
              <button
                key={volume.mount}
                type="button"
                disabled={!fits}
                onClick={() => void choose(volume.suggested)}
                className={`flex w-full items-baseline gap-3 rounded-md px-2.5 py-2 text-left transition-colors ${
                  fits
                    ? "hover:bg-[var(--color-raised)]"
                    : "cursor-not-allowed opacity-45"
                } ${volume.selected ? "bg-[var(--color-raised)]" : ""}`}
              >
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-[13px]">
                    {volume.name}
                    {volume.selected && (
                      <span className="ml-2 text-[11px] text-[var(--color-brass)]">
                        in use
                      </span>
                    )}
                  </span>
                  <span className="figure block text-[11px] text-[var(--color-ink-faint)]">
                    {volume.mount}
                  </span>
                </span>
                <span className="figure shrink-0 text-right text-[12px]">
                  <span className={fits ? "" : "text-[var(--color-over)]"}>
                    {fmt.bytes(volume.free_bytes)}
                  </span>
                  <span className="block text-[11px] text-[var(--color-ink-faint)]">
                    of {fmt.bytes(volume.total_bytes)}
                  </span>
                </span>
              </button>
            );
          })}

          {volumes?.length === 0 && (
            <p className="px-2.5 py-2 text-[12px] text-[var(--color-ink-faint)]">
              No disk with room for a model was found.
            </p>
          )}

          <div className="mt-1 flex flex-wrap gap-2 border-t border-[var(--color-line-soft)] px-2.5 pb-1 pt-2.5">
            <button
              type="button"
              onClick={() => void browse()}
              className="rounded-lg border border-[var(--color-line)] px-2.5 py-1 text-[12px] transition-colors hover:bg-[var(--color-raised)]"
            >
              Choose a folder…
            </button>
            <button
              type="button"
              onClick={() => void choose(null)}
              className="rounded-lg px-2.5 py-1 text-[12px] text-[var(--color-ink-faint)] transition-colors hover:text-[var(--color-ink)]"
            >
              Back to the default
            </button>
          </div>

          {failure && (
            <p
              className="px-2.5 pb-2 text-[12px]"
              style={{ color: "var(--color-over)" }}
            >
              {failure}
            </p>
          )}
        </div>
      )}
    </div>
  );
}
