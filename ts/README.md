# ts/ — TypeScript port of nanochat's inference stack

The goal is not a faster runtime. It is to prove understanding: a port that
produces the *same numbers* as `nanochat/` cannot have been written by
pattern-matching, and a port that does not is wrong in a way you can locate.

## Running it

```bash
# once, and again whenever the reference changes
python -m ts.tools.dump_reference

# the loop you actually live in
node --experimental-strip-types --test ts/test/*.test.ts
```

Node 22.18 strips types natively and ships `node:test`, so there are **no
dependencies at all** — no bundler, no test framework, no tensor library. The
matmul is yours to write; that is the point.

## Layout

| | |
|---|---|
| `tools/dump_reference.py` | PyTorch side: writes the ground truth |
| `src/fixtures.ts` | reads `manifest.json` + `tensors.bin` |
| `src/compare.ts` | the tolerance rule and its reporting |
| `test/plumbing.test.ts` | the harness checking itself |
| `fixtures/` | generated, git-ignored |

## Decisions worth knowing

**The dump is fp32 on CPU, whatever training uses.** Local training runs bf16
on MPS. bf16 carries about three decimal digits, so comparing it against fp32
TypeScript would bury real porting bugs under dtype noise. Parity is checked in
fp32 or not at all.

**Raw f32 + a JSON manifest, not `.npy`/`.safetensors`/JSON numbers.** With no
libraries, whatever gets emitted has to be parsed by hand, and a flat payload
plus a sidecar of offsets is the least code on the reading end. JSON numbers
would lose precision and explode in size. (This layout is roughly what
safetensors does, so the idea transfers.)

**Comparison reports, it does not merely assert.** Ports rarely fail outright —
they drift, and the drift compounds layer by layer. Every check returns its
worst absolute and relative error, so you can watch 1e-7 become 1e-4 across a
stack instead of discovering it at the end.

**Tolerance needs both terms.** `|a - b| <= atol + rtol * |b|`. Absolute alone
is meaningless across values spanning many magnitudes; relative alone explodes
near zero. Presets in `compare.ts`: `ELEMENTWISE` for single ops, `REDUCTION`
after a matmul-sized sum, `DEEP` for the output of a deep stack.

**Every read is checksummed.** The first version of this harness had a hole: a
tensor TypeScript cannot independently derive — a weight matrix, a sample of
`randn` — was only ever compared against itself, so corrupting it changed both
sides equally and every test still passed. Verified by trying it. The manifest
now carries an FNV-1a hash of each tensor's bytes and the reader recomputes it,
which catches truncation, wrong offsets, byte-order flips, and corruption
without needing to know what the values should be.

**Sampling cannot be compared bit for bit.** `torch.multinomial` draws from
PyTorch's RNG, which TypeScript will not reproduce even with a fixed seed. So
parity is checked in three tiers:

| tier | comparable? |
|---|---|
| logits / probability distribution | yes, within tolerance |
| greedy (`argmax`) | yes, token-exact |
| temperature sampling | distribution only — the drawn token is not comparable |

Designing for "same seed, same sentence" would produce a test that cannot pass.

## What this harness deliberately does not do

`test/plumbing.test.ts` touches none of `gpt.py`'s maths — no rotary, no
attention, not even a matmul. The harness has to be trustworthy before it can
judge a port, and every part of it can be exercised on tensors that required no
computation to produce. When a model test later fails, this suite passing is
what tells you the bug is in the model code.
