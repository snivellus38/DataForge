"""
Export a trained checkpoint for the browser.

Layout: one little-endian fp16 blob (`<name>.bin`) plus a JSON manifest (`<name>.json`) giving
each tensor's name, dtype, shape and byte offset. The JS forward pass reads the manifest, slices
the blob, and never needs to know PyTorch existed.

fp16 is safe here: the artifact displays a parallel-vs-recurrent residual, and fp16 round-trip
error on the weights is far below the accumulation error already present in the float32 path.
The exporter VERIFIES this rather than asserting it -- it reports max relative round-trip error
per tensor and the resulting logit drift on a real batch.
"""
import argparse, json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor"))
sys.path.insert(0, os.path.dirname(__file__))
from bdh import BDH, BDHConfig
from cipher_task import make_batch, SRC, TGT, SEP, QRY

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="research/runs/final.pt")
ap.add_argument("--out", default="web/public/model")
a = ap.parse_args()

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = BDHConfig(**{k: v for k, v in ck["cfg"].items()})
model = BDH(cfg); model.load_state_dict(ck["model"]); model.eval()

os.makedirs(os.path.dirname(a.out), exist_ok=True)
# attn.freqs is a DERIVED BUFFER, not a learned parameter, and it must not be shipped as fp16:
# with theta=2**16 its smallest entry is ~2.6e-6, which is SUBNORMAL in fp16 (min normal ~6e-5).
# That was the largest error in the table, on the one tensor where error compounds -- phases are
# position*freq accumulated along the sequence. JS recomputes it exactly from (N, theta) instead.
# Shipped as float32, NOT fp16: smallest entry ~2.6e-6 is fp16-subnormal, and it is the one
# tensor whose error compounds (phases = position*freq along the sequence). Shipping it also
# guarantees the JS phases bit-match PyTorch's float32 get_freqs -- recomputing in JS float64
# would introduce a ~1e-6 phase mismatch that the parity gate should not have to absorb.
FP32 = {"attn.freqs"}
trained = {n for n, _ in model.named_parameters()}

blob, manifest, off = bytearray(), [], 0
for name, p in model.state_dict().items():
    fp32 = name in FP32
    arr = p.detach().cpu().numpy().astype("<f4" if fp32 else "<f2")
    b = arr.tobytes()
    manifest.append({"name": name, "shape": list(arr.shape), "offset": off, "bytes": len(b),
                     "dtype": "float32" if fp32 else "float16", "trained": name in trained,
                     "derived": name in FP32})
    blob += b; off += len(b)
    rt = torch.from_numpy(arr.astype("<f4"))
    # relative-to-RMS, not relative-to-element: elementwise relative error is meaningless on
    # near-zero weights and reports alarming numbers that do not affect the output.
    rms = p.pow(2).mean().sqrt().clamp_min(1e-12)
    print(f"  {name:14s} {str(list(arr.shape)):16s} {len(b):>8,} B  {'fp32' if fp32 else 'fp16'}  "
          f"max err/rms {((rt - p).abs().max() / rms).item():.2e}  trained={name in trained}")

open(a.out + ".bin", "wb").write(bytes(blob))
meta = {
    "arch": "bdh", "source": "github.com/pathwaycom/bdh (MIT), unmodified",
    "weight_tied_across_layers": True,
    "config": {k: v for k, v in ck["cfg"].items()},
    "N": ck["N"], "params": ck["params"],
    "tokens": {"SEP": SEP, "QRY": QRY, "SRC": [ord(c) for c in SRC], "TGT": [ord(c) for c in TGT]},
    "tensors": manifest, "total_bytes": off,
    # Recompute in JS in float64: freqs[i] = 1/(theta**(floor(i/2)*2/N))/(2*pi)
    "rope": {"N": ck["N"], "theta": 2 ** 16, "quantize": 2,
             "note": "derived, not trained; do NOT store as fp16 (smallest entry is fp16-subnormal)"},
    "train": {"acc": {str(k): v for k, v in ck["acc"].items()}, "args": ck["args"]},
}
json.dump(meta, open(a.out + ".json", "w"), indent=1)

# fp16 round-trip drift on a real batch -- the number the README should quote
with torch.no_grad():
    idx, pos, tgt = make_batch(256, 6)
    ref = model(idx)[0][:, pos, :]
    for n_, p in model.named_parameters():
        if n_ not in FP32:
            p.copy_(torch.from_numpy(p.detach().numpy().astype("<f2").astype("<f4")))
    got = model(idx)[0][:, pos, :]
    print(f"\ntotal {off:,} bytes ({off/1024:.0f} KB) fp16")
    print(f"fp16 logit drift: max|delta| = {(ref-got).abs().max().item():.3e}   "
          f"argmax agreement = {(ref.argmax(-1)==got.argmax(-1)).float().mean().item():.2%}")
print(f"wrote {a.out}.bin and {a.out}.json")
