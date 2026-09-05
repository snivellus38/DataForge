"""
Freeze the artifact's demo cases, and VERIFY the ones the narrative depends on.

The sixty-second moment in plan.md is: remove one demonstration, with the test input
byte-identical, and the answer breaks while `parameter updates: 0` never moves. That is an
empirical claim about this checkpoint. It is checked here, not assumed.
"""
import json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor"))
sys.path.insert(0, os.path.dirname(__file__))
from bdh import BDH, BDHConfig
from cipher_task import SRC, TGT, SEP, QRY, seq_len

ck = torch.load("research/runs/final.pt", map_location="cpu", weights_only=False)
model = BDH(BDHConfig(**ck["cfg"])); model.load_state_dict(ck["model"]); model.eval()


def build(pairs, query):
    """pairs: list of (src_char, tgt_char). Returns byte sequence ending at the query symbol."""
    row = []
    for s, t in pairs:
        row += [ord(s), ord(t), SEP]
    row += [QRY, ord(query)]
    return torch.tensor(row, dtype=torch.long).unsqueeze(0)


@torch.no_grad()
def predict(seq, topk=3):
    logits = model(seq)[0][0, -1, :]
    p = torch.softmax(logits, -1)
    v, i = p.topk(topk)
    return [{"byte": int(b), "char": chr(int(b)), "prob": round(float(q), 4)} for q, b in zip(v, i)]


A = [(SRC[0], TGT[0]), (SRC[1], TGT[1]), (SRC[2], TGT[2])]   # 3 demonstrations
QCH = SRC[1]                                                  # query the SECOND one
truth = TGT[1]

full = build(A, QCH)
ablated = build([p for p in A if p[0] != QCH], QCH)            # same query, one demo removed

pf, pa = predict(full), predict(ablated)
ok_full = pf[0]["char"] == truth
broke = pa[0]["char"] != truth
print(f"query {QCH!r} -> truth {truth!r} (byte {ord(truth)})")
print(f"  with    demo: top1={pf[0]['char']!r} p={pf[0]['prob']:.3f}   correct={ok_full}")
print(f"  without demo: top1={pa[0]['char']!r} p={pa[0]['prob']:.3f}   broke={broke}")
print(f"  query bytes identical in both: {torch.equal(full[0, -2:], ablated[0, -2:])}")
print(f"\nSIXTY-SECOND MOMENT {'HOLDS' if (ok_full and broke) else 'DOES NOT HOLD'}")

# does the ablated prediction at least stay in the target alphabet? (structure preserved)
tgt_bytes = {ord(c) for c in TGT}
print(f"  ablated top1 still in target alphabet: {pa[0]['byte'] in tgt_bytes}")

presets = {
    "checkpoint": "research/runs/final.pt",
    "model_acc_by_k": ck["acc"],
    "hook": {
        "why": "opening state: preset already running, no Run button (plan.md 'catchy')",
        "pairs": [[s, t] for s, t in A], "query": QCH, "truth": truth,
        "bytes": full[0].tolist(), "prediction": pf, "correct": ok_full,
    },
    "ablation": {
        "why": "the sixty-second moment: identical query bytes, one demonstration removed",
        "pairs": [[s, t] for s, t in A if s != QCH], "query": QCH, "truth": truth,
        "bytes": ablated[0].tolist(), "prediction": pa, "breaks": broke,
    },
    "verified": {"hook_correct": ok_full, "ablation_breaks": broke},
}
os.makedirs("web/public", exist_ok=True)
json.dump(presets, open("web/public/presets.json", "w"), indent=1)
print("\nwrote web/public/presets.json")
