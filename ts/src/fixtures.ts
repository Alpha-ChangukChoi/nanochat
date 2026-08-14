/**
 * Reader for the reference tensors dumped by ts/tools/dump_reference.py.
 *
 * No dependencies: the manifest is JSON and the payload is raw little-endian
 * f32, so `JSON.parse` and a `Float32Array` view are the entire parser.
 */

import { readFileSync } from "node:fs";
import { join } from "node:path";

export interface TensorEntry {
  name: string;
  dtype: "f32";
  shape: number[];
  offset: number;
  bytes: number;
  /** FNV-1a over the tensor's raw bytes, computed by the dumping side. */
  fnv1a32: number;
}

/**
 * Must match dump_reference.py's fnv1a32. Integer-only on purpose: any float
 * accumulation would depend on summation order and stop agreeing across the
 * two languages.
 */
export function fnv1a32(bytes: Uint8Array): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i];
    // Math.imul keeps the multiply in 32-bit; `h * 0x01000193` would lose the
    // low bits once the product exceeds 2^53.
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

export interface Manifest {
  version: number;
  byte_order: "little";
  payload: string;
  torch_version: string;
  tensors: TensorEntry[];
}

export interface Tensor {
  shape: number[];
  data: Float32Array;
}

export class Fixtures {
  private readonly manifest: Manifest;
  private readonly payload: Buffer;

  constructor(dir: string) {
    this.manifest = JSON.parse(
      readFileSync(join(dir, "manifest.json"), "utf8"),
    ) as Manifest;

    if (this.manifest.version !== 1) {
      throw new Error(`unsupported manifest version ${this.manifest.version}`);
    }
    // Every machine this runs on is little-endian, but the dump records its
    // byte order rather than assuming, so a mismatch is an error and not
    // silently transposed bytes.
    if (this.manifest.byte_order !== "little") {
      throw new Error(`unsupported byte order ${this.manifest.byte_order}`);
    }
    this.payload = readFileSync(join(dir, this.manifest.payload));
  }

  names(): string[] {
    return this.manifest.tensors.map((t) => t.name);
  }

  get(name: string): Tensor {
    const entry = this.manifest.tensors.find((t) => t.name === name);
    if (!entry) {
      throw new Error(
        `no tensor "${name}" in fixtures (have: ${this.names().join(", ")})`,
      );
    }
    const count = entry.shape.reduce((a, b) => a * b, 1);
    if (count * 4 !== entry.bytes) {
      throw new Error(
        `tensor "${name}": shape ${JSON.stringify(entry.shape)} implies ` +
          `${count * 4} bytes but manifest says ${entry.bytes}`,
      );
    }
    if (entry.offset + entry.bytes > this.payload.byteLength) {
      throw new Error(`tensor "${name}" runs past the end of the payload`);
    }
    // byteOffset must be 4-byte aligned for a Float32Array view. Every entry is
    // a whole number of f32s laid end to end, so this holds by construction —
    // assert rather than silently copying, since a violation means the dump
    // format changed.
    const base = this.payload.byteOffset + entry.offset;
    if (base % 4 !== 0) {
      throw new Error(`tensor "${name}" is not 4-byte aligned`);
    }

    // Verified on every read, not only in a dedicated test: a tensor whose
    // values TS cannot derive (a weight matrix, a randn sample) is otherwise
    // only ever compared against itself, so silent corruption would pass.
    const raw = new Uint8Array(this.payload.buffer, base, entry.bytes);
    const got = fnv1a32(raw);
    if (got !== entry.fnv1a32) {
      throw new Error(
        `tensor "${name}" failed its checksum: manifest says ` +
          `0x${entry.fnv1a32.toString(16)}, payload hashes to 0x${got.toString(16)} ` +
          `— fixtures are stale or corrupt; re-run: python -m ts.tools.dump_reference`,
      );
    }

    const data = new Float32Array(this.payload.buffer, base, count);
    return { shape: entry.shape.slice(), data };
  }
}
