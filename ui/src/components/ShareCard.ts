/**
 * The result, as a picture worth posting.
 *
 * People publish their specifications. They have always published their
 * specifications. This draws the answer in a form that survives being pasted
 * into a chat window, which is the only distribution a tool like this gets for
 * free.
 *
 * Drawn rather than screenshotted for three reasons: a screenshot carries
 * whatever else was on screen, it is the size of the window rather than the
 * size of a preview card, and it cannot put the project's name on the result
 * so that someone seeing it knows where it came from.
 *
 * Nothing here recomputes anything. Every figure is one the engine decided,
 * handed in and drawn.
 */
import type { Machine, RankedModel } from "../engine";
import * as fmt from "../format";

/** Card size, in the proportion chat clients preview well. */
const W = 1200;
const PAD = 56;
/** Drawn at twice the size so it stays sharp when a client scales it down. */
const SCALE = 2;

interface Palette {
  ground: string;
  surface: string;
  line: string;
  lineSoft: string;
  ink: string;
  inkDim: string;
  inkFaint: string;
  brass: string;
  weights: string;
  cache: string;
  other: string;
}

/**
 * The theme as the window is actually rendering it.
 *
 * Read from the document rather than restated here, so a card never comes out
 * in colours the application stopped using.
 */
function palette(): Palette {
  const s = getComputedStyle(document.documentElement);
  const v = (name: string) => s.getPropertyValue(name).trim();
  return {
    ground: v("--color-ground"),
    surface: v("--color-surface"),
    line: v("--color-line"),
    lineSoft: v("--color-line-soft"),
    ink: v("--color-ink"),
    inkDim: v("--color-ink-dim"),
    inkFaint: v("--color-ink-faint"),
    brass: v("--color-brass"),
    weights: v("--color-band-weights"),
    cache: v("--color-band-cache"),
    other: v("--color-band-activations"),
  };
}

function rounded(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
): void {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
}

/** Draw one model's bar: the same object the window draws, at card size. */
function bar(
  ctx: CanvasRenderingContext2D,
  p: Palette,
  model: RankedModel,
  x: number,
  y: number,
  width: number,
): void {
  const h = 14;
  const m = model.fit.memory;
  const scale = Math.max(model.fit.pool.usable_bytes, m.required);
  const at = (v: number) => (v / scale) * width;

  ctx.save();
  rounded(ctx, x, y, width, h, h / 2);
  ctx.clip();

  ctx.fillStyle = p.lineSoft;
  ctx.fillRect(x, y, width, h);

  let cursor = x;
  for (const [value, colour] of [
    [m.weights, p.weights],
    [m.kv_cache, p.cache],
    [m.activations + m.runtime_overhead, p.other],
  ] as [number, string][]) {
    const w = at(value);
    ctx.fillStyle = colour;
    ctx.fillRect(cursor, y, w, h);
    cursor += w;
  }

  // Headroom, hatched, because memory left free on purpose is neither used
  // nor available.
  const headroom = at(m.headroom);
  if (headroom > 1) {
    ctx.save();
    ctx.beginPath();
    ctx.rect(cursor, y, headroom, h);
    ctx.clip();
    ctx.strokeStyle = p.line;
    ctx.lineWidth = 2;
    for (let i = -h; i < headroom + h; i += 5) {
      ctx.beginPath();
      ctx.moveTo(cursor + i, y + h);
      ctx.lineTo(cursor + i + h, y);
      ctx.stroke();
    }
    ctx.restore();
  }
  ctx.restore();
}

/** The WhatRunsHere mark, drawn rather than loaded so the card needs no assets. */
function mark(ctx: CanvasRenderingContext2D, p: Palette, x: number, y: number): void {
  const w = 26;
  const h = 30;
  ctx.save();
  ctx.strokeStyle = p.line;
  ctx.lineWidth = 2.5;
  rounded(ctx, x, y, w, h, 5);
  ctx.stroke();

  const bx = x + 5;
  const bw = w - 10;
  ctx.fillStyle = p.weights;
  rounded(ctx, bx, y + 17, bw, 8, 1.5);
  ctx.fill();
  ctx.fillStyle = p.cache;
  rounded(ctx, bx, y + 12, bw, 3.6, 1.5);
  ctx.fill();
  ctx.fillStyle = p.other;
  rounded(ctx, bx, y + 8.8, bw, 2, 1);
  ctx.fill();
  ctx.restore();
}

/**
 * Draw the card and return it as a PNG.
 *
 * Waits on the fonts first. A canvas drawn before they arrive silently falls
 * back to the system face, and the one thing a card like this cannot afford is
 * to look like it came from somewhere else.
 */
export async function render(
  machine: Machine,
  models: RankedModel[],
): Promise<Blob> {
  await document.fonts.ready;

  const p = palette();
  const rows = models.slice(0, 3);
  const width = W - PAD * 2;

  // Laid out first, so the canvas is the height of what it holds. The top
  // row is taller because it carries the plain-language line; repeating that
  // line on every row said the same thing three times and read as filler.
  const headerH = 44;
  const machineH = 96;
  const leadRowH = 104;
  const rowH = 84;
  const footerH = 76;
  const H =
    PAD + headerH + machineH + leadRowH + (rows.length - 1) * rowH + footerH;

  const canvas = document.createElement("canvas");
  canvas.width = W * SCALE;
  canvas.height = H * SCALE;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("this browser has no 2D canvas to draw on");
  ctx.scale(SCALE, SCALE);

  ctx.fillStyle = p.ground;
  ctx.fillRect(0, 0, W, H);

  let y = PAD;

  // ── Header ──────────────────────────────────────────────────────────────
  mark(ctx, p, PAD, y);
  ctx.fillStyle = p.ink;
  ctx.font = "700 21px Archivo, sans-serif";
  ctx.textBaseline = "alphabetic";
  ctx.fillText("WhatRunsHere", PAD + 36, y + 22);

  ctx.fillStyle = p.inkFaint;
  ctx.font = "400 15px 'Instrument Sans', sans-serif";
  ctx.textAlign = "right";
  ctx.fillText("github.com/EuBa-Code/WhatRunsHere", W - PAD, y + 22);
  ctx.textAlign = "left";

  y += headerH;

  // ── The machine ─────────────────────────────────────────────────────────
  ctx.fillStyle = p.inkFaint;
  ctx.font = "600 11px Archivo, sans-serif";
  ctx.fillText("THIS MACHINE RUNS", PAD, y + 14);

  ctx.fillStyle = p.ink;
  ctx.font = "700 34px Archivo, sans-serif";
  ctx.fillText(machine.display.cpu, PAD, y + 54);

  const accelerator = machine.display.accelerators[0];
  ctx.fillStyle = p.inkDim;
  ctx.font = "400 16px 'Instrument Sans', sans-serif";
  ctx.fillText(
    [
      fmt.bytes(machine.detection.system.memory.total_bytes),
      accelerator,
      machine.measurement
        ? `${fmt.bandwidth(machine.calibration.host.bandwidth_bytes_per_s)} measured`
        : null,
    ]
      .filter(Boolean)
      .join("  ·  "),
    PAD,
    y + 80,
  );

  y += machineH;

  // ── The models ──────────────────────────────────────────────────────────
  rows.forEach((model, index) => {
    const lead = index === 0;
    ctx.strokeStyle = p.lineSoft;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(PAD, y);
    ctx.lineTo(W - PAD, y);
    ctx.stroke();

    const top = y + 34;
    ctx.fillStyle = p.ink;
    ctx.font = "600 22px Archivo, sans-serif";
    ctx.fillText(model.name, PAD, top);

    const nameWidth = ctx.measureText(model.name).width;
    ctx.fillStyle = p.inkFaint;
    ctx.font = "400 15px 'IBM Plex Mono', monospace";
    ctx.fillText(
      `${model.fit.quant}  ${fmt.bytes(model.fit.memory.required)}`,
      PAD + nameWidth + 14,
      top,
    );

    // The speed, in the unit and then in words, because tokens per second is
    // a measurement almost nobody has a feel for.
    ctx.textAlign = "right";
    ctx.fillStyle = p.ink;
    ctx.font = "500 26px 'IBM Plex Mono', monospace";
    const speed = fmt.tps(model.fit.decode_tps);
    ctx.fillText(speed, W - PAD - 52, top + 2);
    ctx.fillStyle = p.inkFaint;
    ctx.font = "400 14px 'Instrument Sans', sans-serif";
    ctx.fillText("tok/s", W - PAD, top + 2);
    ctx.textAlign = "left";

    bar(ctx, p, model, PAD, top + 22, width);

    if (lead) {
      ctx.fillStyle = p.inkDim;
      ctx.font = "400 15px 'Instrument Sans', sans-serif";
      ctx.fillText(plainSpeed(model.fit.decode_tps), PAD, top + 60);
    }

    y += lead ? leadRowH : rowH;
  });

  // ── Footer ──────────────────────────────────────────────────────────────
  ctx.strokeStyle = p.line;
  ctx.beginPath();
  ctx.moveTo(PAD, y);
  ctx.lineTo(W - PAD, y);
  ctx.stroke();

  // What the bands are. Without this the picture is pretty and unreadable to
  // anyone who has not opened the application, which is everyone it would
  // reach if it travelled.
  let key = PAD;
  for (const [colour, label] of [
    [p.weights, "weights"],
    [p.cache, "conversation"],
    [p.other, "runtime"],
  ] as [string, string][]) {
    ctx.fillStyle = colour;
    rounded(ctx, key, y + 22, 9, 9, 2);
    ctx.fill();
    ctx.fillStyle = p.inkFaint;
    ctx.font = "400 14px 'Instrument Sans', sans-serif";
    ctx.fillText(label, key + 15, y + 30);
    key += 15 + ctx.measureText(label).width + 22;
  }

  ctx.fillStyle = p.inkFaint;
  ctx.font = "400 14px 'Instrument Sans', sans-serif";
  ctx.textAlign = "right";
  ctx.fillText(
    machine.measurement
      ? "Memory speed timed on this machine, not looked up in a table."
      : "Estimated. Measuring this machine takes a second and sharpens every figure.",
    W - PAD,
    y + 30,
  );
  ctx.textAlign = "left";

  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("the card did not encode"))),
      "image/png",
    );
  });
}

/** Generation speed against the one reference everybody has: reading. */
function plainSpeed(tps: number): string {
  if (tps >= 30) return "Far faster than anyone reads";
  if (tps >= 12) return "Comfortably faster than you read";
  if (tps >= 5) return "About reading pace";
  if (tps >= 2) return "Slower than you read, fine for short answers";
  return "Slow enough that you would leave it running";
}
