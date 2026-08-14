/**
 * The harness checking itself.
 *
 * Nothing here touches gpt.py's maths — no rotary, no attention, no matmul.
 * That is deliberate: the harness has to be trustworthy before it can judge a
 * port, and every part of it (manifest parsing, offsets, alignment, f32
 * round-tripping, the tolerance rule, NaN handling) can be exercised on
 * tensors that required no computation to produce. When a model test later
 * fails, this suite passing is what tells you the bug is in the model code.
 *
 * Fixtures come from:  python -m ts.tools.dump_reference
 */

import { test, describe } from "node:test";
import assert from "node:assert/strict";
import {
  copyFileSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { Fixtures } from "../src/fixtures.ts";
import { compare, assertClose, formatReport, TOLERANCE } from "../src/compare.ts";

const FIXTURE_DIR = join(import.meta.dirname, "..", "fixtures");

if (!existsSync(join(FIXTURE_DIR, "manifest.json"))) {
  throw new Error(
    `no fixtures in ${FIXTURE_DIR} — run:  python -m ts.tools.dump_reference`,
  );
}
const fx = new Fixtures(FIXTURE_DIR);

describe("fixture reader", () => {
  test("shapes survive the round trip", () => {
    assert.deepEqual(fx.get("arange_1d").shape, [16]);
    assert.deepEqual(fx.get("arange_2d").shape, [3, 4]);
    assert.deepEqual(fx.get("arange_3d").shape, [2, 3, 4]);
  });

  test("values land in row-major order", () => {
    // A transposed or column-major reader would still produce the right
    // *multiset* of values here, so check position, not just contents.
    const t = fx.get("arange_2d");
    assert.deepEqual(Array.from(t.data), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    // Non-square, so row-major vs column-major genuinely differ.
    assert.equal(t.data[1 * 4 + 2], 6);
  });

  test("offsets isolate tensors from each other", () => {
    // Every tensor shares one payload buffer; a wrong offset shows up as a
    // neighbour's data bleeding in.
    const a = fx.get("arange_1d");
    const b = fx.get("arange_3d");
    assert.equal(a.data[0], 0);
    assert.equal(a.data[15], 15);
    assert.equal(b.data[0], 0);
    assert.equal(b.data[23], 23);
  });

  test("f32 edge values are bit-exact", () => {
    // These are the values a sloppy format (JSON numbers, f64 casts) mangles.
    const t = fx.get("edge_values");
    assert.equal(t.data[0], 0);
    assert.ok(Object.is(t.data[1], -0), "negative zero must survive");
    assert.equal(t.data[8], 3.4028234663852886e38, "f32 max");
    assert.equal(t.data[9], 1.1754943508222875e-38, "smallest normal");
    assert.ok(t.data[10] > 0, "subnormal must not flush to zero");
    // 0.1 has no exact f32 representation; the point is that both sides agree
    // on *the same* inexact value.
    assert.equal(t.data[11], Math.fround(0.1));
  });

  test("a missing tensor names what is available", () => {
    assert.throws(() => fx.get("nope"), /no tensor "nope"/);
  });

  test("checksums agree with PyTorch's, byte for byte", () => {
    // Without this, tensors TS cannot independently derive — weights, randn —
    // are only ever compared against themselves, and corrupting one would go
    // unnoticed. (It did, the first time this harness was checked.)
    for (const name of fx.names()) {
      assert.doesNotThrow(() => fx.get(name), `checksum failed for ${name}`);
    }
  });

  test("the checksum actually rejects corrupted bytes", () => {
    // A guard nobody has watched fail is not a guard. Flip one bit in a copy of
    // the payload and confirm the reader refuses it.
    const dir = mkdtempSync(join(tmpdir(), "parity-"));
    copyFileSync(join(FIXTURE_DIR, "manifest.json"), join(dir, "manifest.json"));
    const bin = readFileSync(join(FIXTURE_DIR, "tensors.bin"));
    const entry = JSON.parse(
      readFileSync(join(FIXTURE_DIR, "manifest.json"), "utf8"),
    ).tensors.find((t: { name: string }) => t.name === "randn");
    bin[entry.offset + 42 * 4] ^= 0x01; // one bit, in the mantissa's low end
    writeFileSync(join(dir, "tensors.bin"), bin);

    const corrupt = new Fixtures(dir);
    assert.throws(() => corrupt.get("randn"), /failed its checksum/);
    // Neighbouring tensors are untouched and must still read cleanly.
    assert.doesNotThrow(() => corrupt.get("arange_1d"));
    rmSync(dir, { recursive: true, force: true });
  });
});

describe("tolerance rule", () => {
  test("identical data passes at the tightest tolerance", () => {
    const t = fx.get("randn");
    const r = assertClose("randn vs itself", t, t, TOLERANCE.ELEMENTWISE);
    assert.equal(r.maxAbs, 0);
    assert.equal(r.maxRel, 0);
  });

  test("catches a single wrong element", () => {
    const t = fx.get("randn");
    const bad = Float32Array.from(t.data);
    bad[42] = bad[42] + 0.01;
    const r = compare(bad, t, TOLERANCE.REDUCTION);
    assert.equal(r.ok, false);
    assert.equal(r.failures, 1);
    assert.equal(r.worstIndex, 42);
  });

  test("absolute and relative terms cover different failures", () => {
    // Near zero, only atol is meaningful; at large magnitudes, only rtol is.
    // A rule with one term drops one of these two cases.
    const tiny = Float32Array.from([1e-9]);
    const tinyOff = Float32Array.from([1e-9 + 1e-8]);
    assert.equal(compare(tinyOff, tiny, { rtol: 1e-4, atol: 1e-6 }).ok, true,
      "atol should absorb a tiny absolute wobble near zero");
    assert.equal(compare(tinyOff, tiny, { rtol: 1e-4, atol: 0 }).ok, false,
      "with atol=0 the same wobble is a 1000% relative error");

    const big = Float32Array.from([1e6]);
    const bigOff = Float32Array.from([1e6 + 1]);
    assert.equal(compare(bigOff, big, { rtol: 1e-4, atol: 1e-6 }).ok, true,
      "rtol should absorb 1 part in 1e6");
    assert.equal(compare(bigOff, big, { rtol: 0, atol: 1e-6 }).ok, false,
      "with rtol=0 the same wobble is absolutely huge");
  });

  test("NaN and Infinity never compare close", () => {
    const finite = Float32Array.from([1, 2, 3]);
    assert.equal(compare(Float32Array.from([NaN, 2, 3]), finite).ok, false);
    assert.equal(compare(Float32Array.from([Infinity, 2, 3]), finite).ok, false);
    // ...but matching non-finites are fine, so an intentional -Infinity mask
    // (top-k sets rejected logits to -Infinity) does not read as a failure.
    const masked = Float32Array.from([-Infinity, 2, 3]);
    assert.equal(compare(masked, masked).ok, true);
  });

  test("length mismatch fails without pretending to measure error", () => {
    const r = compare(Float32Array.from([1, 2]), Float32Array.from([1, 2, 3]));
    assert.equal(r.ok, false);
    assert.match(r.problems.join(" "), /length 2 != 3/);
  });

  test("report is readable when it fails", () => {
    const r = compare(Float32Array.from([1, 5]), Float32Array.from([1, 2]));
    const text = formatReport("demo", r);
    assert.match(text, /^FAIL demo/);
    assert.match(text, /worst \[1\]: got 5 expected 2/);
  });
});

describe("real weights travel the same path", () => {
  test("a randomly initialised model's weights round-trip", () => {
    // Parity needs *some* weights, not *trained* weights: matching logits under
    // identical weights is the whole claim, and training does not change that.
    const wte = fx.get("wte.weight[:8]");
    assert.equal(wte.shape.length, 2);
    assert.equal(wte.shape[0], 8);
    assert.ok(wte.data.every(Number.isFinite), "weights must all be finite");
    assert.ok(wte.data.some((v) => v !== 0), "weights must not be all zero");
  });

  test("per-layer scalars come through", () => {
    // init_weights sets resid_lambdas to 1 and x0_lambdas to 0; if the dump
    // silently exported the meta-device placeholders instead, this catches it.
    const resid = fx.get("resid_lambdas");
    const x0 = fx.get("x0_lambdas");
    assert.equal(resid.shape[0], 4);
    assert.equal(x0.shape[0], 4);
    assert.ok(resid.data.every(Number.isFinite));
    assert.ok(x0.data.every(Number.isFinite));
  });
});
