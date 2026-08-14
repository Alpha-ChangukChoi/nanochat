/**
 * Numeric comparison for the parity harness.
 *
 * The design choice here is that comparison **reports** rather than only
 * asserting. A port does not usually fail outright — it drifts, and the drift
 * compounds layer by layer. A bare pass/fail hides whether you are at 1e-7 or
 * one bad tolerance bump away from breaking, so every check returns its worst
 * absolute and relative error and where it occurred.
 */

import type { Tensor } from "./fixtures.ts";

export interface Tolerance {
  rtol: number;
  atol: number;
}

/**
 * Both terms are needed. A pure-absolute tolerance is meaningless across a
 * tensor whose values span many orders of magnitude; a pure-relative one blows
 * up near zero, where a difference of 1e-30 reads as 100% error. The combined
 * form `|a - b| <= atol + rtol * |b|` is what numpy/torch use.
 */
export const TOLERANCE = {
  /** Single ops on fp32: differences come only from summation order. */
  ELEMENTWISE: { rtol: 1e-6, atol: 1e-7 } as Tolerance,
  /** One matmul-sized reduction (a few thousand terms). */
  REDUCTION: { rtol: 1e-4, atol: 1e-6 } as Tolerance,
  /** Output of a deep stack, where per-layer error has compounded. */
  DEEP: { rtol: 1e-3, atol: 1e-5 } as Tolerance,
} as const;

export interface Report {
  ok: boolean;
  count: number;
  /** Number of elements outside tolerance. */
  failures: number;
  maxAbs: number;
  maxRel: number;
  /** Flat index of the element with the worst violation (-1 if none). */
  worstIndex: number;
  worstActual: number;
  worstExpected: number;
  /** Shape/length problems and NaN/Inf disagreements, as human-readable lines. */
  problems: string[];
}

function flat(t: Tensor | Float32Array | number[]): Float32Array {
  if (t instanceof Float32Array) return t;
  if (Array.isArray(t)) return Float32Array.from(t);
  return t.data;
}

function shapeOf(t: Tensor | Float32Array | number[]): number[] | null {
  if (t instanceof Float32Array || Array.isArray(t)) return null;
  return t.shape;
}

export function compare(
  actual: Tensor | Float32Array | number[],
  expected: Tensor | Float32Array | number[],
  tol: Tolerance = TOLERANCE.REDUCTION,
): Report {
  const a = flat(actual);
  const b = flat(expected);
  const problems: string[] = [];

  const sa = shapeOf(actual);
  const sb = shapeOf(expected);
  if (sa && sb && JSON.stringify(sa) !== JSON.stringify(sb)) {
    problems.push(`shape ${JSON.stringify(sa)} != ${JSON.stringify(sb)}`);
  }
  if (a.length !== b.length) {
    problems.push(`length ${a.length} != ${b.length}`);
    return {
      ok: false, count: 0, failures: 0, maxAbs: NaN, maxRel: NaN,
      worstIndex: -1, worstActual: NaN, worstExpected: NaN, problems,
    };
  }

  let maxAbs = 0, maxRel = 0, failures = 0;
  let worstIndex = -1, worstScore = -1;
  let worstActual = NaN, worstExpected = NaN;

  for (let i = 0; i < a.length; i++) {
    const x = a[i], y = b[i];

    // NaN and Infinity never compare "close" — they either match exactly or
    // they are a bug. Folding them into the tolerance check would let a NaN
    // slip through as a small difference.
    const xf = Number.isFinite(x), yf = Number.isFinite(y);
    if (!xf || !yf) {
      if (!(Object.is(x, y) || (x === y && xf === yf))) {
        failures++;
        if (problems.length < 8) problems.push(`[${i}] ${x} vs ${y}`);
      }
      continue;
    }

    const abs = Math.abs(x - y);
    const rel = Math.abs(y) > 0 ? abs / Math.abs(y) : abs > 0 ? Infinity : 0;
    if (abs > maxAbs) maxAbs = abs;
    if (Number.isFinite(rel) && rel > maxRel) maxRel = rel;

    const budget = tol.atol + tol.rtol * Math.abs(y);
    if (abs > budget) {
      failures++;
      // Rank by how far past its own budget an element is, so the reported
      // element is the most wrong one rather than merely the largest.
      const score = abs / budget;
      if (score > worstScore) {
        worstScore = score;
        worstIndex = i;
        worstActual = x;
        worstExpected = y;
      }
    }
  }

  return {
    ok: failures === 0 && problems.length === 0,
    count: a.length, failures, maxAbs, maxRel,
    worstIndex, worstActual, worstExpected, problems,
  };
}

export function formatReport(name: string, r: Report): string {
  const head = `${r.ok ? "ok" : "FAIL"} ${name}`;
  const stats =
    `n=${r.count} maxAbs=${r.maxAbs.toExponential(3)} ` +
    `maxRel=${r.maxRel.toExponential(3)}`;
  if (r.ok) return `${head}  ${stats}`;
  const lines = [`${head}  ${stats} failures=${r.failures}`];
  if (r.worstIndex >= 0) {
    lines.push(
      `  worst [${r.worstIndex}]: got ${r.worstActual} expected ${r.worstExpected}`,
    );
  }
  for (const p of r.problems) lines.push(`  ${p}`);
  return lines.join("\n");
}

export function assertClose(
  name: string,
  actual: Tensor | Float32Array | number[],
  expected: Tensor | Float32Array | number[],
  tol: Tolerance = TOLERANCE.REDUCTION,
): Report {
  const r = compare(actual, expected, tol);
  if (!r.ok) throw new Error(formatReport(name, r));
  return r;
}
