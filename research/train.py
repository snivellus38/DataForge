"""
Train the shipping model for the artifact.

This is the script the submission must be able to defend line-by-line, and the one the README's
"how to reproduce" section points at. `feasibility_gate.py` is an experiment harness; this is the
production path. Seeded, checkpointed, and it logs every metric the artifact quotes.

    python research/train.py --steps 30000 --out research/runs/final

Model: the official vendored bdh.py, unmodified. Note it is WEIGHT-TIED across layers --
encoder/encoder_v/decoder are built once and reused in the layer loop -- so `n_layer` is an
iteration count of one shared operator, not depth. Total params = 3*nh*D*N + 2*vocab*D.

Loss: cross-entropy on the ANSWER POSITION ONLY. Demonstration targets are drawn from a fresh
random bijection per example and are therefore unpredictable from context; including them would
add pure noise to the gradient.
"""
import argparse, json, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor"))
sys.path.insert(0, os.path.dirname(__file__))
from bdh import BDH, BDHConfig
from cipher_task import make_batch, SRC, TGT

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=30000)
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--n_embd", type=int, default=64)
ap.add_argument("--n_layer", type=int, default=4)
ap.add_argument("--n_head", type=int, default=2)
ap.add_argument("--mult", type=int, default=8, help="mlp_internal_dim_multiplier; N = mult*n_embd/n_head")
ap.add_argument("--dropout", type=float, default=0.0)
ap.add_argument("--kmin", type=int, default=2)
ap.add_argument("--kmax", type=int, default=12)
ap.add_argument("--eval_every", type=int, default=5000)
ap.add_argument("--out", type=str, default="research/runs/final")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed)

cfg = BDHConfig(n_layer=a.n_layer, n_embd=a.n_embd, n_head=a.n_head,
                mlp_internal_dim_multiplier=a.mult, dropout=a.dropout, vocab_size=256)
model = BDH(cfg).to(dev)
N = a.mult * a.n_embd // a.n_head
nparam = sum(p.numel() for p in model.parameters())
print(f"device={dev}  seed={a.seed}  N={N}  params={nparam:,}  "
      f"sigma={a.n_head}x{N}x{a.n_embd}={a.n_head*N*a.n_embd:,} floats/layer  "
      f"alphabet={len(SRC)}x{len(TGT)} (max k={min(len(SRC),len(TGT))})")

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
ce = torch.nn.functional.cross_entropy


@torch.no_grad()
def evaluate(ks, n=512, chunk=64):
    """Chunked -- a wide batch at large k OOMs (x_sparse is B*nh*T*N floats)."""
    model.eval()
    out = {}
    for k in ks:
        hits = 0
        for i in range(0, n, chunk):
            b = min(chunk, n - i)
            idx, pos, tgt = make_batch(b, k, dev)
            hits += (model(idx)[0][:, pos, :].argmax(-1) == tgt).sum().item()
        out[k] = hits / n
    model.train()
    return out


EVAL_KS = [2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48]
hist, t0 = [], time.time()
for step in range(1, a.steps + 1):
    k = int(torch.randint(a.kmin, a.kmax + 1, (1,)).item())
    idx, pos, tgt = make_batch(a.batch, k, dev)
    loss = ce(model(idx)[0][:, pos, :], tgt)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 1000 == 0 or step == 1:
        hist.append({"step": step, "loss": loss.item(), "t": round(time.time() - t0, 1)})
        print(f"  step {step:6d}  loss {loss.item():.4f}  ({time.time()-t0:.0f}s)")
    if step % a.eval_every == 0 or step == a.steps:
        acc = evaluate(EVAL_KS)
        print("    acc " + "  ".join(f"k{k}={v:.0%}" for k, v in acc.items()))

acc = evaluate(EVAL_KS, n=2048)
os.makedirs(os.path.dirname(a.out), exist_ok=True)
torch.save({"model": model.state_dict(), "cfg": vars(cfg), "args": vars(a),
            "N": N, "params": nparam, "acc": acc}, a.out + ".pt")
json.dump({"args": vars(a), "N": N, "params": nparam, "acc": {str(k): v for k, v in acc.items()},
           "history": hist, "device": dev,
           "repro": f"python research/train.py --steps {a.steps} --seed {a.seed} --mult {a.mult} --out {a.out}"},
          open(a.out + ".json", "w"), indent=1)
print(f"\nfinal (n=2048/k): " + "  ".join(f"k{k}={v:.1%}" for k, v in acc.items()))
print(f"wrote {a.out}.pt and {a.out}.json")
