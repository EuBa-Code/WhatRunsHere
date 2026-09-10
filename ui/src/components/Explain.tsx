/**
 * Saying what a term means, where the term is.
 *
 * This application is dense on purpose, and dense is only a virtue if someone
 * can get in. The way in is not a simplified mode — that would put the real
 * product behind a switch — but an explanation attached to the word that needs
 * it, available on hover and on focus, and absent otherwise.
 *
 * Native `title` attributes were what this replaced. They take a second to
 * appear, cannot be reached from the keyboard, and are drawn by the operating
 * system in a style nothing else here shares — which is a poor way to deliver
 * the one idea the application is built around.
 */
import { useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

/** Distance kept from the window edges when an explanation would overflow. */
const MARGIN = 12;
/** Width of the panel, matching the `w-64` it is drawn at. */
const WIDTH = 256;

interface Placement {
  left: number;
  top: number;
  above: boolean;
}

/**
 * A term with an explanation attached.
 *
 * A word gains a faint dotted underline, the long-standing convention for
 * "there is more here". **A figure does not**, and the distinction is not
 * cosmetic: a dotted rule under a numeral already means something here — that
 * nothing measured it — and a second dotted line meaning "hover me" would make
 * the one claim this product rests on unreadable. Around a figure the
 * affordance appears on hover, where nothing is being asserted.
 */
export function Explain({
  children,
  title,
  body,
  className = "",
  underline = true,
}: {
  children: ReactNode;
  /** The term, restated. Omit when the trigger is already the term. */
  title?: string;
  body: ReactNode;
  className?: string;
  /** False around anything carrying a provenance rule. See above. */
  underline?: boolean;
}) {
  const id = useId();
  const [placement, setPlacement] = useState<Placement | null>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const panel = useRef<HTMLDivElement>(null);

  // Positioned against the window and drawn at the end of the document,
  // because every ancestor here is a candidate for clipping it: the view
  // scrolls, the cards round their corners with `overflow: hidden`, and a term
  // near the left edge would push a centred panel off the screen. Fixed
  // coordinates in a portal are subject to none of that.
  const place = () => {
    const box = trigger.current?.getBoundingClientRect();
    if (!box) return;
    setPlacement({
      left: clamp(
        box.left + box.width / 2 - WIDTH / 2,
        MARGIN,
        window.innerWidth - WIDTH - MARGIN,
      ),
      top: box.top,
      above: box.top > 220,
    });
  };

  // Re-checked once the panel has a height, so a long explanation opening
  // upwards flips down rather than running off the top of the window.
  useLayoutEffect(() => {
    if (!placement || !placement.above || !panel.current) return;
    if (placement.top - panel.current.offsetHeight - 8 < MARGIN) {
      setPlacement({ ...placement, above: false });
    }
  }, [placement]);

  const box = trigger.current?.getBoundingClientRect();

  return (
    <>
      <button
        ref={trigger}
        type="button"
        aria-describedby={placement ? id : undefined}
        onMouseEnter={place}
        onMouseLeave={() => setPlacement(null)}
        onFocus={place}
        onBlur={() => setPlacement(null)}
        // The explanation is the whole point of the control, so a click that
        // does nothing else should not also dismiss it.
        onClick={(event) => event.preventDefault()}
        className={`pointer-events-auto cursor-help transition-colors ${
          underline
            ? "underline decoration-dotted decoration-[var(--color-line)] underline-offset-[3px] hover:decoration-[var(--color-ink-faint)]"
            : "hover:text-[var(--color-ink)]"
        } ${className}`}
      >
        {children}
      </button>
      {placement &&
        createPortal(
          <div
            ref={panel}
            id={id}
            role="tooltip"
            className="pointer-events-none fixed z-50 w-64 rounded-lg border border-[var(--color-line)] bg-[var(--color-raised)] px-3 py-2.5 text-left font-sans text-[12px] leading-relaxed font-normal normal-case tracking-normal text-[var(--color-ink-dim)] shadow-2xl"
            style={{
              left: placement.left,
              top: placement.above
                ? placement.top - 8
                : (box?.bottom ?? placement.top) + 8,
              transform: placement.above ? "translateY(-100%)" : undefined,
              animation: "explain-in 110ms ease-out",
            }}
          >
            {title && (
              <p className="mb-1 font-medium text-[var(--color-ink)]">{title}</p>
            )}
            {body}
          </div>,
          document.body,
        )}
    </>
  );
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(Math.max(value, low), Math.max(low, high));
}

/**
 * What the quantization formats mean, in the terms someone choosing between
 * them would want.
 *
 * Deliberately about the trade rather than the algorithm: the block structure
 * of a K-quant is not what anybody is deciding between.
 */
export function quantExplanation(quant: string, bitsPerWeight: number): {
  title: string;
  body: string;
} {
  const bits = `${bitsPerWeight.toFixed(1)} bits per weight`;
  const name = quant.toUpperCase();

  if (name.startsWith("F16") || name.startsWith("BF16") || name.startsWith("F32")) {
    return {
      title: `${quant} — no compression`,
      body: `The weights exactly as trained, at ${bits}. Nothing is lost and nothing is saved: this is the largest the model gets, and the answers it gives are the ones its makers measured.`,
    };
  }
  const severity =
    bitsPerWeight >= 6
      ? "Barely distinguishable from the original."
      : bitsPerWeight >= 4.5
        ? "The format most people run: a large saving for a small, usually unnoticeable cost."
        : bitsPerWeight >= 3.5
          ? "A real saving with a cost you may notice on harder questions."
          : "Aggressive. It fits where nothing else does, and it shows.";
  return {
    title: `${quant} — compressed weights`,
    body: `Each weight stored in about ${bits} instead of 16. The file shrinks by roughly the same ratio, and so does the memory it needs and the time spent reading it. ${severity}`,
  };
}

/** What a quality score is a score *of*, and where this one sits. */
export function qualityExplanation(rank: { at_or_below: number; of: number } | null): {
  title: string;
  body: string;
} {
  const placing = rank
    ? ` Among the ${rank.of} models here, it scores at least as well as ${rank.at_or_below - 1} of the other ${rank.of - 1}.`
    : "";
  return {
    title: "Quality, 0–100",
    body: `A weighted average of this model's published benchmark results — MMLU-Pro, GPQA, LiveCodeBench, MATH and others — weighted for what you said you would use it for. Nothing scores near 100: the benchmarks are hard, and the best models here are in the sixties.${placing}`,
  };
}

/**
 * A generation speed, against the only reference everyone has.
 *
 * Adult silent reading runs around 240 words a minute, and a word is a little
 * over a token, so a comfortable reading pace is roughly five tokens a second.
 * That is the line between "waiting" and "keeping up".
 */
export function speedExplanation(tps: number): { title: string; body: string } {
  const comparison =
    tps >= 30
      ? "Far faster than anyone reads. It will feel instant."
      : tps >= 12
        ? "Comfortably faster than you read. Text arrives ahead of your eyes."
        : tps >= 5
          ? "About reading pace. You will be reading as it writes, with no waiting."
          : tps >= 2
            ? "Slower than you read. Usable for short answers, tiring for long ones."
            : "Slower than a conversation tolerates. Fine for a batch job left running, painful to sit in front of.";
  return {
    title: `${tps < 10 ? tps.toFixed(1) : Math.round(tps)} tokens per second`,
    body: `A token is about three quarters of a word, and most people read around five tokens a second. ${comparison}`,
  };
}
