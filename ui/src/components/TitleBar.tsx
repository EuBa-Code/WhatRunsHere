/**
 * The window's own title bar.
 *
 * The system one is a separate grey band above the application, with its own
 * typeface and its own idea of how tall a bar should be. Drawing it here puts
 * the navigation, the status and the window controls on one line in one
 * typeface, which is most of the difference between an application and a web
 * page in a frame.
 *
 * What that costs has to be paid back deliberately, because the system was
 * doing all of it: the bar must be draggable, double-clicking it must maximise,
 * and the three controls must exist and behave the way every other window on
 * the platform does, including the close button turning red, which is the one
 * piece of platform styling nobody expects to lose.
 */
import { useEffect, useState, type ReactNode } from "react";
import { getCurrentWindow } from "@tauri-apps/api/window";

export function TitleBar({ children }: { children: ReactNode }) {
  const [maximized, setMaximized] = useState(false);
  const appWindow = getCurrentWindow();

  useEffect(() => {
    let live = true;
    const sync = () =>
      void appWindow.isMaximized().then((value) => live && setMaximized(value));
    void sync();
    // The window can be maximised by a snap gesture or a keyboard shortcut,
    // neither of which passes through the button below.
    const unlisten = appWindow.onResized(sync);
    return () => {
      live = false;
      void unlisten.then((stop) => stop());
    };
  }, [appWindow]);

  return (
    <header
      // The whole bar drags, except the controls inside it, which opt out.
      data-tauri-drag-region
      className="flex h-11 shrink-0 select-none items-center border-b border-[var(--color-line)] bg-[var(--color-surface)] pl-4"
    >
      {/* The bar's contents take the whole width and place themselves; the
          controls sit after them and never move, so nothing in the middle can
          leave a gap in front of the close button. */}
      <div
        data-tauri-drag-region
        className="flex min-w-0 flex-1 items-center gap-6"
      >
        {children}
      </div>
      <div className="flex h-full shrink-0">
        <Control onClick={() => void appWindow.minimize()} label="Minimise">
          <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden>
            <rect x="0" y="4.5" width="10" height="1" fill="currentColor" />
          </svg>
        </Control>
        <Control
          onClick={() => void appWindow.toggleMaximize()}
          label={maximized ? "Restore" : "Maximise"}
        >
          {maximized ? (
            <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden>
              <rect
                x="0.5"
                y="2.5"
                width="7"
                height="7"
                fill="none"
                stroke="currentColor"
              />
              <path d="M2.5 2.5V0.5H9.5V7.5H7.5" fill="none" stroke="currentColor" />
            </svg>
          ) : (
            <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden>
              <rect
                x="0.5"
                y="0.5"
                width="9"
                height="9"
                fill="none"
                stroke="currentColor"
              />
            </svg>
          )}
        </Control>
        <Control onClick={() => void appWindow.close()} label="Close" danger>
          <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden>
            <path d="M0 0L10 10M10 0L0 10" stroke="currentColor" fill="none" />
          </svg>
        </Control>
      </div>
    </header>
  );
}

function Control({
  children,
  onClick,
  label,
  danger = false,
}: {
  children: ReactNode;
  onClick: () => void;
  label: string;
  danger?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      title={label}
      className={`flex h-full w-[46px] items-center justify-center text-[var(--color-ink-dim)] transition-colors ${
        danger
          ? "hover:bg-[#c42b1c] hover:text-white"
          : "hover:bg-[var(--color-raised)] hover:text-[var(--color-ink)]"
      }`}
    >
      {children}
    </button>
  );
}
