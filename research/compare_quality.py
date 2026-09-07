#!/usr/bin/env python
"""Bits per byte, on the same bytes, against public models 15x to 70x larger.

WHY THIS SCRIPT EXISTS
----------------------
The 8M model's loss has always been quoted as 0.670 nats/byte, which is a number with no context
attached. Nats/byte is also not comparable to anything published, because every other model reports
loss per TOKEN under its own tokenizer. Converting to bits per byte -- 0.670 / ln 2 = 0.967 --
makes it comparable to any language model whatsoever, because a byte is a byte.

So: measure our 8M byte-level BDH and several public pretrained transformers on exactly the same
text, with exactly the same windowing, and normalise every one of them by the same byte count.

WHAT THIS IS NOT
----------------
It is NOT a controlled comparison and the output must never be quoted as one. These are public
models trained by other people on other data at other scales, with different tokenizers and
different objectives. Nothing here isolates the effect of an architecture. What it does give is
CONTEXT: a reader who does not know whether 0.97 bits/byte is good can see what a 124M and a 560M
model score on the same text.

THE TWO CONFOUNDS, BOTH NAMED, RUNNING IN OPPOSITE DIRECTIONS
-------------------------------------------------------------
1. SPECIALISATION, in our favour. Our model was trained on Europarl in the exact interleaved
   `<F:en>...<T:fr>...` format the in-domain set uses. The public models have never seen that
   format and are not specialists in parliamentary proceedings. On this set we should win, and a
   win here says our model fits this distribution well -- not that it is better at language.

2. CONTAMINATION, in theirs. Europarl v7 is old, public, and widely scraped, so it is plausibly
   inside the pretraining data of the public models. Nobody can check, so it is stated.

Which is why there are TWO evaluation sets. The out-of-domain set is French prose from a different
century and a different genre, in plain form with no tags. We expect to lose that one badly, and
the size of the gap between the two sets is the honest measure of how specialised a small
specialist really is.

WINDOWING, AND WHY IT IS BY CHARACTER AND NOT BY TOKEN
------------------------------------------------------
Every model sees the same 512-byte-ish window and is scored on the second half of it, so each
scored byte has roughly 256 bytes of context and no model gets a longer context than another. The
text is cut on CHARACTER boundaries (never mid-UTF-8-sequence), and for the tokenizer models the
split point is mapped to a token index with the tokenizer's own offset mapping, so the scored
region is exactly the same span of text for everyone. The denominator is always the UTF-8 byte
length of that span.

Our model is also held to its trained 512-byte context here, not to the 4,096 rows its positional
embedding happens to have.

USAGE
    python research/compare_quality.py                       # needs data/en-fr/val.bin
    python research/compare_quality.py --skip-hf             # our model only, no downloads
"""
import argparse
import contextlib
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from bdh_big import load_big  # noqa: E402

# Public models, fixed before any of them was run. Parameter counts are read from the loaded model,
# not typed, so the table cannot drift from what was actually measured.
BASELINES = [
    ("gpt2", "the canonical reference point"),
    ("HuggingFaceTB/SmolLM2-135M", "a modern small model, so the frontier is not all 2019"),
    ("bigscience/bloom-560m", "multilingual, genuinely trained on French"),
]

OOD_URL = "https://www.gutenberg.org/cache/epub/4650/pg4650.txt"   # Candide, French, public domain


def load_indomain(path, n_chars, seed=0, blocks=20):
    """Held-out Europarl, rebuilt by the notebook's own 95/5 split of the byte stream.

    Sampled from offsets spread across the WHOLE split, not one contiguous chunk. The first
    version read the first 60 KB and got 1.13 bits/byte where the training run's own validation
    reports 0.967 -- a single region of a parliamentary corpus is not representative of it, and
    that gap was sampling, not the model. Each block starts at a record boundary.
    """
    if not os.path.exists(path):
        return None
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    rng = np.random.default_rng(seed)
    per = n_chars // blocks
    starts = rng.integers(0, max(len(raw) - 4 * per, 1), size=blocks)
    out = []
    for st in sorted(starts):
        chunk = bytes(raw[st: st + 4 * per])
        i = chunk.find(b"<F:en>")           # never begin mid-record
        if i < 0:
            continue
        out.append(chunk[i:].decode("utf-8", errors="ignore")[:per])
    return out


def load_ood(cache, n_chars):
    """French prose from a different century and genre, in plain form. Public domain."""
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        import urllib.request
        req = urllib.request.Request(OOD_URL, headers={"User-Agent": "DataForge-eval/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                open(cache, "wb").write(r.read())
        except Exception as e:
            print("  could not fetch the out-of-domain text (%s)" % e)
            return None
    txt = open(cache, encoding="utf-8", errors="ignore").read()
    # Take only what is between Gutenberg's own markers, then drop the front matter and table of
    # contents. Anchoring on the first "CHAPITRE" instead lands in the TOC, which is a list of
    # headings rather than prose -- the first version of this scored 10.04 bits/byte on it.
    a = txt.find("*** START OF")
    b = txt.find("*** END OF")
    body = txt[txt.find("\n", a) + 1: b if b > a else len(txt)]
    body = body[6000:]

    # Two normalisations, both disclosed in the output, so that this set measures GENRE and not
    # character set. The training corpus was read as latin-1 and contains no typographic quotes,
    # and its lines are whole sentences rather than wrapped at 70 columns. Left as-is, most of the
    # OOD penalty would be the model meeting U+2019 for the first time, which is not the question.
    for a_, b_ in [("’", "'"), ("‘", "'"), ("“", '"'), ("”", '"'),
                   ("—", "--"), ("–", "-"), ("œ", "oe"), ("Œ", "OE")]:
        body = body.replace(a_, b_)
    paras = [" ".join(p.split()) for p in body.split("\n\n")]
    return " ".join(p for p in paras if p)[:n_chars]


def windows(blocks, win_chars):
    """Cut on character boundaries into (context, scored) halves.

    `blocks` is a list of independent passages, and windows NEVER span two of them. The in-domain
    set is sampled from scattered offsets, so a window straddling a seam would carry context from
    an unrelated part of the corpus into its scored half -- genuinely harder, and an artifact of
    the sampling rather than a property of the model. That flaw cost 0.06 bits/byte when it was in.
    """
    if isinstance(blocks, str):
        blocks = [blocks]
    out = []
    for text in blocks:
        for i in range(0, len(text) - win_chars + 1, win_chars):
            w = text[i:i + win_chars]
            out.append((w[: win_chars // 2], w[win_chars // 2:]))
    return out


def autocast_ctx(device, precision):
    """One numerical regime for every model in the table. See PRECISION in the module docstring."""
    if precision == "fp32" or device != "cuda":
        return contextlib.nullcontext()
    return torch.autocast("cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)


@torch.no_grad()
def bpb_bdh(model, wins, device, precision="bf16"):
    """Our byte model: NLL over the scored half, divided by that half's byte count."""
    nats = 0.0
    nbytes = 0
    amp = autocast_ctx(device, precision)
    for ctx, scored in wins:
        ids = list((ctx + scored).encode("utf-8"))
        lo = len(ctx.encode("utf-8"))
        if len(ids) > 512 or lo < 1:
            continue
        with amp:
            logits, _ = model(torch.tensor([ids], dtype=torch.long, device=device))
        lp = F.log_softmax(logits[0].float(), -1)
        tgt = torch.tensor(ids[lo:], device=device)
        nats += float(-lp[lo - 1:len(ids) - 1].gather(1, tgt.unsqueeze(1)).sum())
        nbytes += len(ids) - lo
    return nats / max(nbytes, 1) / math.log(2), nbytes


@torch.no_grad()
def bpb_hf(name, wins, device, precision="bf16"):
    """A tokenizer model, scored on exactly the same span of text and divided by the same bytes.

    Weights load in fp32 and the forward runs under the SAME autocast as ours, so precision is not
    a free variable between rows of the table."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    lm = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).to(device).eval()
    amp = autocast_ctx(device, precision)
    nats = 0.0
    nbytes = 0
    for ctx, scored in wins:
        text = ctx + scored
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        offs = enc["offset_mapping"][0].tolist()
        if ids.shape[1] < 4 or ids.shape[1] > getattr(lm.config, "max_position_embeddings", 4096):
            continue
        # First token whose span starts at or after the context/scored boundary.
        start = next((i for i, (a, _) in enumerate(offs) if a >= len(ctx)), None)
        if start is None or start < 1 or start >= ids.shape[1]:
            continue
        with amp:
            out_logits = lm(ids).logits
        lp = F.log_softmax(out_logits[0].float(), -1)
        tgt = ids[0, start:]
        nats += float(-lp[start - 1:-1].gather(1, tgt.unsqueeze(1)).sum())
        # The scored region is the text those tokens actually cover, so the denominator matches.
        nbytes += len(text[offs[start][0]:].encode("utf-8"))
    params = sum(p.numel() for p in lm.parameters())
    del lm
    if device == "cuda":
        torch.cuda.empty_cache()
    return nats / max(nbytes, 1) / math.log(2), nbytes, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "models/checkpoints/french_best.pt"))
    ap.add_argument("--val", default=os.path.join(ROOT, "data/en-fr/val.bin"))
    ap.add_argument("--ood-cache", default=os.path.join(ROOT, "data/ood/fr-prose.txt"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chars", type=int, default=60000, help="characters sampled per evaluation set")
    ap.add_argument("--win", type=int, default=480, help="window in characters; half is scored")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--precision", default="bf16", choices=["bf16", "fp16", "fp32"],
                    help="one autocast regime applied to every model in the table")
    ap.add_argument("--skip-hf", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "research/runs/quality.json"))
    ap.add_argument("--png", default=os.path.join(ROOT, "docs/media/efficiency-frontier.png"))
    args = ap.parse_args()

    sets = {}
    ind = load_indomain(args.val, args.chars, seed=args.seed)
    if ind:
        sets["in-domain (held-out Europarl)"] = windows(ind, args.win)
    else:
        print("  no %s -- rebuild it with the notebook's Cell 3 downloader"
              % os.path.relpath(args.val, ROOT))
    ood = load_ood(args.ood_cache, args.chars)
    if ood:
        sets["out-of-domain (French prose)"] = windows(ood, args.win)
    if not sets:
        return 1

    model, cfg, _ = load_big(args.ckpt, device=args.device)
    rows = {}
    ours = "8M BDH (ours)"
    rows[ours] = {"params": int(sum(p.numel() for p in model.parameters())), "bpb": {}}
    for sname, wins in sets.items():
        b, nb = bpb_bdh(model, wins, args.device, args.precision)
        rows[ours]["bpb"][sname] = round(b, 4)
        rows[ours].setdefault("scored_bytes", {})[sname] = nb
        print("  %-28s %-32s %.4f bits/byte  (%s bytes)" % (ours, sname, b, f"{nb:,}"))
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    if not args.skip_hf:
        for name, why in BASELINES:
            rows[name] = {"why_in_the_set": why, "bpb": {}}
            for sname, wins in sets.items():
                try:
                    b, nb, params = bpb_hf(name, wins, args.device, args.precision)
                except Exception as e:
                    print("  %-28s FAILED: %s" % (name, e))
                    break
                rows[name]["params"] = params
                rows[name]["bpb"][sname] = round(b, 4)
                rows[name].setdefault("scored_bytes", {})[sname] = nb
                print("  %-28s %-32s %.4f bits/byte  (%s bytes)" % (name, sname, b, f"{nb:,}"))

    out = {
        "precision": args.precision,
        "window_chars": args.win,
        "sets": list(sets.keys()),
        "not_a_controlled_comparison": (
            "Public models trained by other people on other data at other scales. This gives "
            "context for our number, not evidence about architectures. Two confounds run in "
            "opposite directions: our model is a specialist on the in-domain set, and Europarl v7 "
            "is old and public so it may be inside their pretraining data."),
        "models": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)
    print("\n  wrote", os.path.relpath(args.out, ROOT))

    try:
        plot(rows, list(sets.keys()), args.png, args.precision)
        print("  wrote", os.path.relpath(args.png, ROOT))
    except Exception as e:
        print("  (no figure: %s)" % e)
    return 0


def plot(rows, sets, path, precision="bf16"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    ind, ood = sets[0], (sets[1] if len(sets) > 1 else None)

    # One vertical connector per model. Its LENGTH is the generalisation gap, which is the whole
    # point of running two sets, and it also stops the two series from needing two labels each.
    for name, r in rows.items():
        b = r.get("bpb", {})
        if ood and ind in b and ood in b and r.get("params"):
            ax.plot([r["params"], r["params"]], [b[ind], b[ood]], "-", lw=1.2, color="0.72",
                    zorder=1)

    for sname, marker, colour in zip(sets, ("o", "s"), ("#2b6ca8", "#d98032")):
        xs = [r["params"] for r in rows.values() if sname in r.get("bpb", {}) and r.get("params")]
        ys = [r["bpb"][sname] for r in rows.values() if sname in r.get("bpb", {}) and r.get("params")]
        ax.plot(xs, ys, marker, ms=9, color=colour, label=sname, zorder=3)

    for name, r in rows.items():
        b = r.get("bpb", {})
        if ind in b and r.get("params"):
            ax.annotate(name.split("/")[-1], (r["params"], b[ind]), fontsize=8,
                        xytext=(7, -3), textcoords="offset points", zorder=4)

    ours = rows.get("8M BDH (ours)", {})
    ob = ours.get("bpb", {})
    if ood and ind in ob and ood in ob:
        ax.annotate("specialisation gap", (ours["params"], (ob[ind] + ob[ood]) / 2),
                    fontsize=8, color="0.4", xytext=(10, 0), textcoords="offset points",
                    va="center")

    xs_all = [r["params"] for r in rows.values() if r.get("params")]
    ax.set_xscale("log")
    ax.set_xlim(min(xs_all) / 2.4, max(xs_all) * 4.0)
    ax.set_xlabel("parameters")
    ax.set_ylabel("bits per byte (lower is better)")
    ax.set_title("Same bytes, same windows, same %s regime, one script" % precision, fontsize=11)
    ax.grid(alpha=0.25, which="both", lw=0.5)
    ax.legend(frameon=False, fontsize=9, loc="upper center")
    fig.text(0.5, -0.04, "Not a controlled comparison: different data, different scales, different "
                         "tokenizers. Context for our number, not evidence about architectures.",
             ha="center", fontsize=7.5, color="0.35")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    raise SystemExit(main())
