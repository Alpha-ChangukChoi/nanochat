"""
Dump reference tensors from PyTorch for the TypeScript parity harness.

The TS side re-implements nanochat's inference stack; this script produces the
ground truth it is checked against. Run it once to (re)generate fixtures:

    python -m ts.tools.dump_reference

Two things about this script are load-bearing:

1. It forces **fp32 on CPU**, regardless of what the training config uses. The
   local training standard is bf16 on MPS (see the runcpu findings), and bf16
   carries roughly three decimal digits — comparing that against fp32 TS would
   bury real porting bugs under dtype noise. Parity is checked in fp32 or not
   at all.

2. It writes **raw little-endian f32 + a JSON manifest**, not .npy/.safetensors
   /JSON-numbers. The TS side may not use libraries, so we have to parse
   whatever we emit; a flat payload plus a sidecar of offsets is the least code
   on the reading end. (That is essentially safetensors' layout, so the shape of
   the idea transfers.)
"""

import argparse
import json
import os
import struct

# Force the parity dtype/device before anything reads the env (nanochat.common
# resolves COMPUTE_DTYPE at import time).
os.environ["NANOCHAT_DTYPE"] = "float32"

import torch


def fnv1a32(data: bytes) -> int:
    """FNV-1a over the raw bytes, as an integer the TS side can reproduce exactly.

    A checksum is what makes tensors verifiable that TS cannot *derive*. A weight
    matrix or a sample of randn has no independently computable expectation, so
    a reader that silently returned the wrong bytes for it would go unnoticed —
    every check on such a tensor ends up comparing the fixture against itself.
    Byte-level hashing sidesteps that: it catches truncation, a wrong offset, a
    byte-order flip, and corruption, without needing to know what the values
    should be. It is deliberately integer-only, since any float accumulation
    would depend on summation order and stop matching across the two languages.
    """
    h = 0x811C9DC5
    for b in data:
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return h


def dump(tensors: dict[str, torch.Tensor], out_dir: str) -> str:
    """Write {name: tensor} as one .bin payload plus a manifest describing it."""
    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, "tensors.bin")
    manifest_path = os.path.join(out_dir, "manifest.json")

    entries = []
    offset = 0
    with open(bin_path, "wb") as f:
        for name, t in tensors.items():
            # .contiguous() so the byte order matches the logical shape; the TS
            # reader assumes plain row-major with no strides.
            a = t.detach().to(torch.float32).contiguous().cpu()
            buf = a.numpy().tobytes()
            f.write(buf)
            entries.append({
                "name": name,
                "dtype": "f32",
                "shape": list(a.shape),
                "offset": offset,
                "bytes": len(buf),
                "fnv1a32": fnv1a32(buf),
            })
            offset += len(buf)

    manifest = {
        "version": 1,
        "byte_order": "little",
        "payload": "tensors.bin",
        "torch_version": torch.__version__,
        "tensors": entries,
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def plumbing_tensors() -> dict[str, torch.Tensor]:
    """Tensors that exercise the harness without exercising any model math.

    The harness has to be trustworthy before it can judge a port, so its own
    first test involves no attention, no rotary, nothing from gpt.py — only
    values that are awkward to round-trip. If these fail, the bug is in the
    plumbing, and knowing that is the whole point.
    """
    g = torch.Generator().manual_seed(1337)
    return {
        # Shape handling: 1-D, 2-D, 3-D, and a non-square 2-D so a transposed
        # reader fails loudly instead of silently passing.
        "arange_1d": torch.arange(16, dtype=torch.float32),
        "arange_2d": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "arange_3d": torch.arange(24, dtype=torch.float32).reshape(2, 3, 4),
        # Value handling: signs, zeros, and magnitudes far apart enough that a
        # pure-absolute or pure-relative tolerance alone would be wrong.
        "randn": torch.randn(64, 8, generator=g),
        "edge_values": torch.tensor([
            0.0, -0.0, 1.0, -1.0,
            1e-8, -1e-8, 1e8, -1e8,
            3.4028234663852886e38,   # f32 max
            1.1754943508222875e-38,  # f32 smallest normal
            1e-45,                   # subnormal
            0.1, 0.2, 0.30000001192092896, 1.0 / 3.0, 2.0 / 3.0,
        ], dtype=torch.float32),
    }


def weight_tensors() -> dict[str, torch.Tensor]:
    """A real (small, randomly initialised) nanochat model's weights.

    Proves the same path carries actual parameters, and pins the decision that
    the parity reference needs *some* weights rather than *trained* weights —
    matching logits under identical weights is what parity means, and training
    changes nothing about that.
    """
    from nanochat.gpt import GPT, GPTConfig

    torch.manual_seed(1337)
    cfg = GPTConfig(sequence_len=256, vocab_size=32768, n_layer=4,
                    n_head=4, n_kv_head=4, n_embd=256, window_pattern="L")
    model = GPT(cfg)
    model.init_weights()
    return {
        "wte.weight[:8]": model.transformer.wte.weight[:8],
        "h0.attn.c_q.weight[:4,:8]": model.transformer.h[0].attn.c_q.weight[:4, :8],
        "resid_lambdas": model.resid_lambdas,
        "x0_lambdas": model.x0_lambdas,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=os.path.join("ts", "fixtures"),
                   help="directory to write tensors.bin + manifest.json into")
    p.add_argument("--no-weights", action="store_true",
                   help="skip the model weights (plumbing tensors only)")
    args = p.parse_args()

    tensors = plumbing_tensors()
    if not args.no_weights:
        tensors.update(weight_tensors())

    assert struct.calcsize("f") == 4  # sanity: the manifest promises 4-byte f32
    path = dump(tensors, args.out)
    print(f"wrote {len(tensors)} tensors -> {path}")
    for name, t in tensors.items():
        print(f"  {name:<28} {tuple(t.shape)}")


if __name__ == "__main__":
    main()
