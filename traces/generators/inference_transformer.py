#!/usr/bin/env python3
"""Deterministic JSONL trace generator for poolkit.

Models a transformer inference workload with bimodal lifetime, which is the one
property a log-uniform request density (the cost model's placeholder) gets
wrong:

  * weights: few, large, allocated once and freed at the very end (long-lived);
  * activations: many, small, allocated and freed within a single step
    (short-lived).

Output, side by side in the destination directory:
  * <name>.jsonl       the trace, one op per line (op,size,ptr,stream,ordinal);
  * <name>.hist.txt    the histogram of allocation sizes the cost model is
                       measured against.

Everything is a pure function of (layers, hidden, batch, steps, seed): the same
arguments reproduce the exact same bytes, which is what makes the replay test
"run twice, get identical counters" meaningful.

Sizes are kept strictly within [256, 2 MiB] so every request is served by the
pool (no large path), and the internal-waste comparison against the model is
clean. float32 element size is assumed.
"""

import argparse
import collections
import json
import random
import sys
from pathlib import Path

BYTES_PER_ELEMENT = 4  # float32


def jittered(rng: random.Random, base: int, frac: float) -> int:
    """Deterministic multiplicative jitter around `base`, in [base*(1-frac),
    base*(1+frac)]. Exercises several size classes instead of one."""
    factor = 1.0 + frac * (2.0 * rng.random() - 1.0)
    return max(256, int(base * factor))


class TraceWriter:
    def __init__(self, path: Path):
        self.path = path
        self.handle = path.open("w", encoding="ascii", newline="\n")
        self.ordinal = 0
        self.ptr = 0
        self.histogram = collections.Counter()

    def alloc(self, size: int, stream: int = 0) -> int:
        ptr = self.ptr
        self.ptr += 1
        self._write({"op": "alloc", "size": size, "ptr": ptr, "stream": stream,
                     "ordinal": self.ordinal})
        self.histogram[size] += 1
        return ptr

    def free(self, ptr: int, stream: int = 0) -> None:
        self._write({"op": "free", "size": 0, "ptr": ptr, "stream": stream,
                     "ordinal": self.ordinal})

    def _write(self, op: dict) -> None:
        self.handle.write(json.dumps(op, separators=(",", ":")) + "\n")
        self.ordinal += 1

    def close(self) -> None:
        self.handle.close()


def generate(layers: int, hidden: int, batch: int, steps: int, seed: int,
             out_dir: Path, name: str) -> tuple[Path, Path]:
    rng = random.Random(seed)
    jsonl_path = out_dir / f"{name}.jsonl"
    hist_path = out_dir / f"{name}.hist.txt"

    writer = TraceWriter(jsonl_path)

    weight_bytes = hidden * hidden * BYTES_PER_ELEMENT          # ~1 MiB @512
    activation_bytes = hidden * BYTES_PER_ELEMENT               # ~2 KiB @512
    intermediate_bytes = 4 * hidden * BYTES_PER_ELEMENT         # ~8 KiB @512

    # --- weights: long-lived, freed at the end ------------------------------
    # 2 embeddings + per layer Q/K/V/O + MLP-in/MLP-out.
    weight_ptrs = []
    for _ in range(2):
        weight_ptrs.append(writer.alloc(jittered(rng, weight_bytes, 0.05)))
    for _ in range(layers):
        for _ in range(6):
            weight_ptrs.append(writer.alloc(jittered(rng, weight_bytes, 0.05)))

    # --- activations: short-lived, churned every step -----------------------
    for _ in range(steps):
        for _ in range(layers):
            # one intermediate (MLP hidden) per layer, short-lived
            writer.free(writer.alloc(jittered(rng, intermediate_bytes, 0.10)))
            for _ in range(batch):
                # one hidden vector per token, very short-lived
                writer.free(writer.alloc(jittered(rng, activation_bytes, 0.15)))

    # --- release the weights at the end -------------------------------------
    for ptr in weight_ptrs:
        writer.free(ptr)
    writer.close()

    # --- histogram ----------------------------------------------------------
    # Binned by log2(size), one line per bin: "lo..hi  count". The model
    # integrates internal waste over these sizes (docs/cost-model.md §2).
    bins = collections.Counter()
    for size, count in sorted(writer.histogram.items()):
        bins[size.bit_length()] += count
    with hist_path.open("w", encoding="ascii", newline="\n") as out:
        out.write("# allocation-size histogram (bin = [2^(b-1), 2^b) bytes)\n")
        for b in sorted(bins):
            out.write(f"{1 << (b - 1) if b > 0 else 0:>12} .. {1 << b:>12}  {bins[b]}\n")
        out.write(f"# total allocs: {sum(writer.histogram.values())}\n")
        out.write(f"# distinct sizes: {len(writer.histogram)}\n")

    return jsonl_path, hist_path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=6, help="transformer layers")
    parser.add_argument("--hidden", type=int, default=512, help="hidden dimension")
    parser.add_argument("--batch", type=int, default=4, help="tokens per step")
    parser.add_argument("--steps", type=int, default=50, help="forward steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=Path("traces"))
    parser.add_argument("--name", default="inference_transformer")
    args = parser.parse_args(argv)

    if args.layers < 1 or args.hidden < 1 or args.batch < 1 or args.steps < 1:
        parser.error("layers/hidden/batch/steps must be >= 1")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path, hist_path = generate(
        args.layers, args.hidden, args.batch, args.steps, args.seed,
        args.out_dir, args.name,
    )
    print(f"wrote {jsonl_path}")
    print(f"wrote {hist_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
