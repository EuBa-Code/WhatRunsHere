/**
 * The pieces every view is built from.
 *
 * [`Figure`] is the important one. Every number the engine produces knows how
 * it was arrived at, and that status is drawn as a rule under the numeral:
 * solid for something measured here, dashed for a manufacturer's figure,
 * dotted for an assumption. One legend explains the three; after that the page
 * can be read without labels on every value.
 */
import type { CSSProperties, ReactNode } from "react";
import type { Confidence } from "../engine";
import * as fmt from "../format";

/** A number, with a rule beneath it saying where it came from. */
export function Figure({
  value,
  unit,
  confidence,
  size = "base",
}: {
  value: string;
  unit?: string;
  /** Omit for a figure that is simply a fact, like a file size. */
  confidence?: Confidence;
  size?: "base" | "large" | "hero";
}) {
  const scale = {
    base: "text-[15px]",
    large: "text-[26px] leading-none",
    hero: "text-[40px] leading-none",
  }[size];
  const prov = confidence ? fmt.provenance(confidence) : null;
  const quiet = size !== "base" ? "prov-quiet" : "";

  return (
    <span
      className="inline-flex items-baseline gap-1.5"
      title={prov ? `${prov.label}. ${prov.detail}` : undefined}
    >
      <span
        className={`figure font-medium ${scale} ${prov?.className ?? ""} ${
          prov ? quiet : ""
        }`}
      >
        {value}
      </span>
      {unit && (
        <span className="text-[12px] text-[var(--color-ink-faint)]">{unit}</span>
      )}
    </span>
  );
}

/** The legend that makes the rules readable. Shown once, in the status bar. */
export function ProvenanceLegend() {
  const entries: { className: string; label: string }[] = [
    { className: "prov prov-measured", label: "measured" },
    { className: "prov prov-spec", label: "from spec" },
    { className: "prov prov-assumed", label: "assumed" },
  ];
  return (
    <div
      className="hidden items-center gap-3 md:flex"
      title="Every figure carries a rule saying where it came from."
    >
      {entries.map((entry) => (
        <span
          key={entry.label}
          className="flex items-center gap-1.5 text-[10px] text-[var(--color-ink-faint)]"
        >
          <span
            className={`${entry.className} prov-quiet inline-block h-2 w-4`}
            aria-hidden
          />
          {entry.label}
        </span>
      ))}
    </div>
  );
}

export function Card({
  children,
  className = "",
  style,
}: {
  children: ReactNode;
  className?: string;
  /** Carries the `animation-delay` that staggers a view's arrival. */
  style?: CSSProperties;
}) {
  return (
    <section className={`card ${className}`} style={style}>
      {children}
    </section>
  );
}

/** A small capitalised label that names a region without shouting. */
export function Eyebrow({ children }: { children: ReactNode }) {
  return <p className="eyebrow">{children}</p>;
}

/** A verdict, coloured by what it says. */
export function VerdictPill({
  label,
  tone,
}: {
  label: string;
  tone: string;
}) {
  const colour = `var(--color-${tone})`;
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium"
      style={{ color: colour, background: `color-mix(in oklab, ${colour} 13%, transparent)` }}
    >
      <span
        className="h-1.5 w-1.5 rounded-full"
        style={{ background: colour }}
        aria-hidden
      />
      {label}
    </span>
  );
}

/** A labelled row in a specification list. */
export function Spec({
  label,
  children,
}: {
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="flex items-baseline justify-between gap-6 border-b border-[var(--color-line-soft)] py-2.5 last:border-0">
      <dt className="text-[13px] text-[var(--color-ink-dim)]">{label}</dt>
      <dd className="text-right">{children}</dd>
    </div>
  );
}

/**
 * A choice among a handful of options, as a segmented control.
 *
 * A select would be smaller, but these are the controls that change the whole
 * answer (what the model is for, what to optimise for), and hiding them
 * behind a closed menu hides that the answer depends on them.
 */
export function Segmented<T extends string>({
  value,
  options,
  onChange,
  label,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
  label: string;
}) {
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className="inline-flex rounded-lg border border-[var(--color-line)] bg-[var(--color-ground)] p-0.5"
    >
      {options.map((option) => {
        const active = option.value === value;
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={active}
            onClick={() => onChange(option.value)}
            className={`rounded-[6px] px-2.5 py-1 text-[12px] transition-colors ${
              active
                ? "bg-[var(--color-raised)] text-[var(--color-ink)]"
                : "text-[var(--color-ink-faint)] hover:text-[var(--color-ink-dim)]"
            }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}

/** A horizontal 0 to 100 meter, for a bounded score. */
export function Meter({ value, tone = "brass" }: { value: number; tone?: string }) {
  return (
    <div
      className="h-1 w-full overflow-hidden rounded-full"
      style={{ background: "var(--color-line-soft)" }}
    >
      <div
        className="h-full rounded-full"
        style={{
          width: `${Math.max(0, Math.min(100, value))}%`,
          background: `var(--color-${tone})`,
        }}
      />
    </div>
  );
}

/** Nothing to show, and what to do about it. */
export function Empty({ title, children }: { title: string; children: ReactNode }) {
  return (
    <div className="card flex flex-col items-center gap-2 px-8 py-16 text-center">
      <p className="font-display text-[17px] font-semibold">{title}</p>
      <p className="max-w-md text-[13px] leading-relaxed text-[var(--color-ink-dim)]">
        {children}
      </p>
    </div>
  );
}
