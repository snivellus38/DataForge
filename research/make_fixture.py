"""
Golden outputs for the JS parity gate (plan.md Phase 2).

Critical detail: the reference is computed from the FP16-ROUNDED weights, i.e. exactly what the
browser loads. Comparing JS against the full-precision checkpoint would fold a ~6e-3 quantisation
gap into the gate and make a correct port look broken.

Also dumps per-stage intermediates so a failing port can be bisected by layer instead of guessed at.
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor"))
sys.path.insert(0, os.path.dirname(__file__))
from bdh import BDH, BDHConfig
from cipher_task import make_batch

ck = torch.load("research/runs/final.pt", map_location="cpu", weights_only=False)
model = BDH(BDHConfig(**ck["cfg"])); model.load_state_dict(ck["model"]); model.eval()
FP32 = {"attn.freqs"}
with torch.no_grad():                      # match the exported blob exactly
    for n, p in model.named_parameters():
        if n not in FP32:
            p.copy_(torch.from_numpy(p.detach().numpy().astype("<f2").astype("<f4")))

taps = {}
def tap(name):
    def hook(_m, _i, o): taps.setdefault(name, o.detach().clone())
    return hook
model.attn.register_forward_hook(tap("attn_out_layer0"))

torch.manual_seed(123)
cases = []
for k in (3, 6, 10):
    idx, pos, ans = make_batch(2, k)
    taps.clear()
    with torch.no_grad():
        logits = model(idx)[0]
    cases.append({
        "k": k, "answer_pos": int(pos),
        "idx": idx.tolist(),
        "logits_last": logits[:, -1, :].flatten().tolist(),      # (B, vocab) at final position
        "logits_answer": logits[:, pos, :].flatten().tolist(),   # (B, vocab) at the answer position
        "argmax_answer": logits[:, pos, :].argmax(-1).tolist(),
        "attn_out_layer0": taps["attn_out_layer0"].flatten().tolist(),
        "attn_out_shape": list(taps["attn_out_layer0"].shape),
    })
    print(f"  k={k}  T={idx.shape[1]}  argmax@answer={logits[:,pos,:].argmax(-1).tolist()}  "
          f"truth={ans.tolist()}")

os.makedirs("web/test", exist_ok=True)
json.dump({"note": "reference computed from FP16-ROUNDED weights == web/public/model.bin",
           "config": ck["cfg"], "tolerance": {"logits": 1e-4, "attn": 1e-4}, "cases": cases},
          open("web/test/fixture.json", "w"))
print(f"\nwrote web/test/fixture.json ({os.path.getsize('web/test/fixture.json')/1024:.0f} KB)")
