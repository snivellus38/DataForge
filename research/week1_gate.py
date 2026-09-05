"""
WEEK-1 FEASIBILITY GATE (plan.md): does a deliberately shrunk BDH learn the in-context
substitution cipher, and where is the capacity cliff in k (simultaneous bindings)?

Everything downstream of the plan depends on a YES here. Run before writing any frontend.
"""
import argparse, sys, time, os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "vendor"))
sys.path.insert(0, os.path.dirname(__file__))
from bdh import BDH, BDHConfig
from cipher_task import make_batch

p = argparse.ArgumentParser()
p.add_argument("--steps", type=int, default=4000)
p.add_argument("--batch", type=int, default=256)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--n_embd", type=int, default=64)
p.add_argument("--n_layer", type=int, default=4)
p.add_argument("--n_head", type=int, default=2)
p.add_argument("--mult", type=int, default=8)
p.add_argument("--kmin", type=int, default=2)
p.add_argument("--kmax", type=int, default=10)
p.add_argument("--keval", type=int, default=14)
p.add_argument("--out", type=str, default="")
p.add_argument("--ckpt", type=str, default="")
a = p.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

cfg = BDHConfig(n_layer=a.n_layer, n_embd=a.n_embd, n_head=a.n_head,
                mlp_internal_dim_multiplier=a.mult, dropout=0.0, vocab_size=256)
model = BDH(cfg).to(dev)
N = a.mult * a.n_embd // a.n_head
nparam = sum(q.numel() for q in model.parameters())
print(f"device={dev}  N={N} neurons/head  params={nparam:,}  "
      f"sigma={a.n_head}x{N}x{a.n_embd}={a.n_head*N*a.n_embd:,} floats/layer")

opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.1)
lossf = torch.nn.functional.cross_entropy

t0 = time.time()
for step in range(1, a.steps + 1):
    k = int(torch.randint(a.kmin, a.kmax + 1, (1,)).item())
    idx, pos, tgt = make_batch(a.batch, k, dev)
    logits, _ = model(idx)
    loss = lossf(logits[:, pos, :], tgt)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if step % 500 == 0 or step == 1:
        print(f"  step {step:5d}  loss {loss.item():.4f}  ({time.time()-t0:.0f}s)")

@torch.no_grad()
def acc(k, n=512, chunk=64):
    """Chunked: a 512-wide eval batch at large k OOMs (x_sparse is B*nh*T*N floats)."""
    model.eval()
    hits = 0
    for i in range(0, n, chunk):
        b = min(chunk, n - i)
        idx, pos, tgt = make_batch(b, k, dev)
        hits += (model(idx)[0][:, pos, :].argmax(-1) == tgt).sum().item()
    model.train()
    return hits / n

print(f"\ntrained on k in [{a.kmin},{a.kmax}]   (n=512 eval per k)")
print("  k :  acc   | in train range")
rows = []
for k in range(2, a.keval + 1, max(1, (a.keval - 2) // 12)):
    v = acc(k); rows.append((k, v))
    print(f"  {k:2d} : {v:6.1%} | {'yes' if a.kmin <= k <= a.kmax else 'NO (extrapolation)'}")

if a.ckpt:
    torch.save({"model": model.state_dict(), "cfg": vars(cfg), "args": vars(a)}, a.ckpt)
    print("wrote", a.ckpt)

if a.out:
    import json
    json.dump({"config": vars(a), "N": N, "params": nparam, "acc": rows},
              open(a.out, "w"), indent=1)
    print("wrote", a.out)
